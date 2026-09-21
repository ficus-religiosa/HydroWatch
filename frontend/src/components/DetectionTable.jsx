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
              <th>Confidence</th>
              <th>Size</th>
              <th>Location</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td>{row.id}</td>
                <td>{row.label}</td>
                <td>{row.confidence}%</td>
                <td>{row.size}</td>
                <td>{row.x}px, {row.y}px</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DetectionTable;
