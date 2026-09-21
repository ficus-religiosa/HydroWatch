function StatCard({ label, value, change, tone = 'blue' }) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <p>{label}</p>
        <span className={`badge badge-${tone}`}>{change}</span>
      </div>
      <h3>{value}</h3>
    </div>
  );
}

export default StatCard;
