function LoadingState({ text = 'Loading data...' }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      <span>{text}</span>
    </div>
  );
}

export default LoadingState;
