function DetectionTable({ rows }) {
  return (
    <div className="panel">
      <div className="section-heading">
        <h3>Detected objects</h3>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Class</th>
              <th>Type</th>
              <th>Confidence</th>
              <th>Size</th>
              <th>Position (x, y)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.id}</td>
                <td>{row.label}</td>
                <td>{row.tier || 'debris'}</td>
                <td>{row.confidence}%</td>
                <td>{row.size}</td>
                <td>{row.x.toFixed(1)}%, {row.y.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!rows.length && <p className="analysis-note">No objects match the current filter in this frame.</p>}
    </div>
  );
}

export default DetectionTable;
