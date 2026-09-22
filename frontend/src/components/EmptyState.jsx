function EmptyState({ title, body }) {
  return (
    <div className="empty-state">
      <h4>{title}</h4>
      <p>{body}</p>
    </div>
  );
}

export default EmptyState;
