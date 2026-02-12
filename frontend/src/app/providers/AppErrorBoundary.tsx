import { Component, type ErrorInfo, type PropsWithChildren } from 'react';

interface AppErrorBoundaryState {
  hasError: boolean;
}

export class AppErrorBoundary extends Component<PropsWithChildren, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError(): AppErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Keep diagnostics in the console for local debugging.
    console.error('Unhandled app error', error, errorInfo);
  }

  private onReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <main className="fatal-error">
        <div className="fatal-error__card">
          <h1>Something went wrong</h1>
          <p>The app hit an unexpected error. Reload to continue.</p>
          <button type="button" onClick={this.onReload}>
            Reload App
          </button>
        </div>
      </main>
    );
  }
}
