package runner

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"sync"
	"time"

	"rules_itest/logger"
	"rules_itest/runner/topological"
	"rules_itest/svclib"
)

// We need to use process groups to reliably tear down services and their descendants.
// This is especially important in hot-reload mode, where you need to restart the child
// and have it bind the same port.
// However, we don't want to do this in tests, because Bazel will already terminate the
// test process (svcinit) and all its children.
// If we were to start new process groups in tests, we could leak children (at least on Mac).
var shouldUseProcessGroups = runtime.GOOS != "windows" && os.Getenv("BAZEL_TEST") != "1"
var terseOutput = os.Getenv("SVCINIT_TERSE_OUTPUT") == "True"

type ServiceSpecs = map[string]svclib.VersionedServiceSpec

// ReuseportListenFn produces a fresh SO_REUSEPORT-aware listener bound to addr.
// Injected so the runner stays platform-agnostic (the actual SO_REUSEPORT
// sockopt setup lives in cmd/svcinit).
type ReuseportListenFn func(addr string) (net.Listener, error)

type Runner struct {
	ctx          context.Context
	serviceSpecs ServiceSpecs

	serviceInstances map[string]*ServiceInstance

	// portsMu guards portListeners. StartAll releases listeners from the
	// per-service worker goroutines (topological runner, parallel), so even
	// though writes to different keys, the map itself needs serialization.
	portsMu sync.Mutex
	// portListeners holds open SO_REUSEPORT listeners keyed by service label.
	// On startup the runner inherits listeners from the bootstrap port-assignment
	// pass; they are closed once the service becomes healthy, releasing the
	// port back to be exclusively owned by the service. Across an ibazel
	// restart of a SO_REUSEPORT-aware service we rebind on the same addresses
	// before stopping the old instance, so the port can't be stolen during the
	// gap between Stop and Start (the "baton pass").
	portListeners map[string][]net.Listener

	// reuseportAddrs preserves the addresses originally bound for each
	// SO_REUSEPORT-aware service so the runner can rebind across a restart
	// even after the original listeners were released. Written once in New;
	// read-only thereafter.
	reuseportAddrs map[string][]string

	reuseportListenFn ReuseportListenFn
}

func New(
	ctx context.Context,
	serviceSpecs ServiceSpecs,
	portListeners map[string][]net.Listener,
	reuseportListenFn ReuseportListenFn,
) (*Runner, error) {
	addrs := make(map[string][]string, len(portListeners))
	for label, ls := range portListeners {
		for _, l := range ls {
			addrs[label] = append(addrs[label], l.Addr().String())
		}
	}
	r := &Runner{
		ctx:               ctx,
		serviceInstances:  map[string]*ServiceInstance{},
		portListeners:     portListeners,
		reuseportAddrs:    addrs,
		reuseportListenFn: reuseportListenFn,
	}
	err := r.UpdateSpecs(serviceSpecs, nil)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func colorize(s svclib.VersionedServiceSpec) string {
	return s.Colorize(s.Label)
}

func (r *Runner) StartAll(serviceErrCh chan error) ([]topological.Task, error) {
	tasks := allTasks(r.serviceInstances, func(ctx context.Context, service *ServiceInstance) error {
		if service.Type == "group" {
			return nil
		}

		if service.Deferred {
			log.Printf("Deferring %s\n", colorize(service.VersionedServiceSpec))
			return nil
		}

		if terseOutput {
			log.Printf("Starting %s\n", colorize(service.VersionedServiceSpec))
		} else {
			log.Printf("Starting %s %v\n", colorize(service.VersionedServiceSpec), service.cmd.Args[1:])
		}

		startErr := service.Start(ctx)
		if startErr != nil {
			return startErr
		}

		go func() {
			err := service.Wait()
			if err != nil && !service.Killed() {
				serviceErrCh <- fmt.Errorf(colorize(service.VersionedServiceSpec) + " exited with error: " + err.Error())
			}
		}()

		if service.VersionedServiceSpec.HealthCheckTimeout != "" {
			timeout, err := time.ParseDuration(service.VersionedServiceSpec.HealthCheckTimeout)
			if err != nil {
				log.Printf("failed to parse health check timeout, falling back to no timeout: %v", err)
			}
			timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
			ctx = timeoutCtx
			defer cancel()
		}
		if err := service.WaitUntilHealthy(ctx); err != nil {
			return err
		}
		r.releasePortListeners(service.Label)
		return nil
	})
	starter := topological.NewRunner(tasks)
	err := starter.Run(r.ctx)

	return starter.CriticalPath(), err
}

func (r *Runner) StopAll() (map[string]*os.ProcessState, error) {
	tasks := allTasks(r.serviceInstances, func(ctx context.Context, service *ServiceInstance) error {
		if service.Type == "group" || service.Deferred {
			return nil
		}
		log.Printf("Stopping %s\n", colorize(service.VersionedServiceSpec))
		return service.Stop()
	})
	stopper := topological.NewReversedRunner(tasks)
	err := stopper.Run(r.ctx)

	states := make(map[string]*os.ProcessState)

	for _, serviceInstance := range r.serviceInstances {
		if serviceInstance.Type == "group" || serviceInstance.Deferred {
			continue
		}
		states[serviceInstance.Label] = serviceInstance.ProcessState()
	}

	return states, err
}

func (r *Runner) GetStartDurations() map[string]time.Duration {
	durations := make(map[string]time.Duration)

	for _, serviceInstance := range r.serviceInstances {
		durations[serviceInstance.Label] = serviceInstance.startDuration
	}

	return durations
}

func (r *Runner) GetInstance(label string) *ServiceInstance {
	return r.serviceInstances[label]
}

type updateActions struct {
	toStopLabels   []string
	toStartLabels  []string
	toReloadLabels []string
}

func computeUpdateActions(currentServices, newServices ServiceSpecs) updateActions {
	actions := updateActions{}

	// Check if existing services need a reload, a restart, or a shutdown.
	for label, service := range currentServices {
		newService, ok := newServices[label]
		if !ok {
			fmt.Println(label + " has been removed, stopping")
			actions.toStopLabels = append(actions.toStopLabels, label)
			continue
		}

		// We technically don't need a restart if the change is the list of deps.
		// But that should not be a common use case, so it's not worth the complexity.
		if !reflect.DeepEqual(service, newService) {
			log.Printf(colorize(service) + " definition or code has changed, restarting...")
			if service.HotReloadable && reflect.DeepEqual(service.ServiceSpec, newService.ServiceSpec) {
				// The only difference is the Version. Trust the service that
				// it prefers to receive the ibazel reload command.
				actions.toReloadLabels = append(actions.toReloadLabels, label)
			} else {
				actions.toStopLabels = append(actions.toStopLabels, label)
				actions.toStartLabels = append(actions.toStartLabels, label)
			}
			continue
		}
	}

	// Handle new services
	for label := range newServices {
		if _, ok := currentServices[label]; !ok {
			actions.toStartLabels = append(actions.toStartLabels, label)
		}
	}

	return actions
}

func (r *Runner) UpdateSpecs(serviceSpecs ServiceSpecs, ibazelCmd []byte) error {
	updateActions := computeUpdateActions(r.serviceSpecs, serviceSpecs)

	willRestart := make(map[string]struct{}, len(updateActions.toStartLabels))
	for _, label := range updateActions.toStartLabels {
		willRestart[label] = struct{}{}
	}

	for _, label := range updateActions.toStopLabels {
		serviceInstance := r.serviceInstances[label]
		if serviceInstance.Type == "group" {
			continue
		}
		// Baton-pass: if this service is being restarted (not removed) and we
		// already released its SO_REUSEPORT listeners after a previous healthy,
		// rebind on the original addresses now -- before stopping the old
		// process -- so the port can't be stolen during the restart gap.
		if _, ok := willRestart[label]; ok {
			r.rebindPortListeners(label)
		}
		serviceInstance.Stop()
		delete(r.serviceInstances, label)
	}

	for _, label := range updateActions.toStartLabels {
		var err error
		r.serviceInstances[label], err = prepareServiceInstance(r.ctx, serviceSpecs[label])
		if err != nil {
			return err
		}
	}

	for _, label := range updateActions.toReloadLabels {
		_, err := r.serviceInstances[label].stdin.Write(ibazelCmd)
		if err != nil {
			return err
		}
	}

	r.serviceSpecs = serviceSpecs
	return nil
}

// releasePortListeners closes any SO_REUSEPORT listeners the runner is holding
// for label and clears the entry. Safe to call when nothing is held.
func (r *Runner) releasePortListeners(label string) {
	r.portsMu.Lock()
	listeners := r.portListeners[label]
	delete(r.portListeners, label)
	r.portsMu.Unlock()

	for _, l := range listeners {
		if err := l.Close(); err != nil {
			log.Printf("Warning: failed to close port listener for %s: %v", label, err)
		}
	}
}

// rebindPortListeners reopens SO_REUSEPORT listeners on the originally-bound
// addresses for label, but only if we no longer hold them (i.e. they were
// released after the prior healthy). No-op when there's nothing to rebind --
// e.g. non-SO_REUSEPORT services, or restarts where the previous boot never
// reached healthy and we are still holding the listeners.
func (r *Runner) rebindPortListeners(label string) {
	r.portsMu.Lock()
	alreadyHeld := len(r.portListeners[label]) > 0
	r.portsMu.Unlock()
	if alreadyHeld {
		return
	}
	addrs := r.reuseportAddrs[label]
	if len(addrs) == 0 || r.reuseportListenFn == nil {
		return
	}
	listeners := make([]net.Listener, 0, len(addrs))
	for _, addr := range addrs {
		l, err := r.reuseportListenFn(addr)
		if err != nil {
			// Most likely some non-SO_REUSEPORT process snuck in and grabbed
			// the port during the gap since release. Warn and continue with
			// whatever we managed to rebind -- this is no worse than today.
			log.Printf("Warning: failed to rebind port listener for %s on %s: %v", label, addr, err)
			continue
		}
		listeners = append(listeners, l)
	}
	if len(listeners) > 0 {
		r.portsMu.Lock()
		r.portListeners[label] = listeners
		r.portsMu.Unlock()
	}
}

func (r *Runner) UpdateSpecsAndRestart(
	serviceSpecs ServiceSpecs,
	serviceErrCh chan error,
	ibazelCmd []byte,
) (
	[]topological.Task, error,
) {
	err := r.UpdateSpecs(serviceSpecs, ibazelCmd)
	if err != nil {
		return nil, err
	}
	return r.StartAll(serviceErrCh)
}

func prepareServiceInstance(ctx context.Context, s svclib.VersionedServiceSpec) (*ServiceInstance, error) {
	if s.Type == "group" {
		return &ServiceInstance{
			VersionedServiceSpec: s,
			startErrFn:           sync.OnceValue(func() error { return nil }),
		}, nil
	}

	instance := &ServiceInstance{
		VersionedServiceSpec: s,
	}

	err := initializeServiceCmd(ctx, instance)
	if err != nil {
		return nil, err
	}

	return instance, nil
}

func initializeServiceCmd(ctx context.Context, instance *ServiceInstance) error {
	s := instance.VersionedServiceSpec

	cmd := exec.CommandContext(ctx, s.Exe, s.Args...)
	// Note, this leaks the caller's env into the service, so it's not hermetic.
	// For `bazel test`, Bazel is already sanitizing the env, so it's fine.
	// For `bazel run`, there is no expectation of hermeticity, and it can be nice to use env to control behavior.
	cmd.Env = os.Environ()
	for k, v := range s.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stdout = logger.New(s.Label+"> ", s.Color, os.Stdout)
	cmd.Stderr = logger.New(s.Label+"> ", s.Color, os.Stderr)

	if shouldUseProcessGroups {
		setPgid(cmd)
	}

	// Even if a child process exits, Wait will block until the I/O pipes are closed.
	// They may have been forwarded to an orphaned child, so we disable that behavior to unblock exit.
	if s.Type == "service" && !s.Deferred {
		// We need a bit of grace period to allow I/O pipes to close on our end.
		cmd.WaitDelay = 50 * time.Millisecond
	}

	instance.cmd = cmd
	instance.killed = false
	instance.startErrFn = sync.OnceValue(cmd.Start)
	instance.waitErrFn = sync.OnceValue(func() error {
		res := cmd.Wait()
		instance.SetDone()
		return res
	})

	if s.HotReloadable {
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return err
		}
		instance.stdin = stdin
	}

	instance.mu.Lock()
	instance.runErr = nil
	instance.done = false
	instance.healthcheckAttempted = false
	instance.mu.Unlock()

	return nil
}
