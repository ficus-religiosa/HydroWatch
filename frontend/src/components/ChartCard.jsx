import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area, Legend, LineChart, Line } from 'recharts';

function ChartCard({ title, type = 'bar', data, dataKeys, color = '#5ec3ff' }) {
  const renderChart = () => {
    if (type === 'area') {
      return (
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.35} />
                <stop offset="95%" stopColor={color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#203754" />
            <XAxis dataKey="name" stroke="#9bb0c8" />
            <YAxis stroke="#9bb0c8" />
            <Tooltip />
            <Area type="monotone" dataKey={dataKeys[0]} stroke={color} fill="url(#areaFill)" />
          </AreaChart>
        </ResponsiveContainer>
      );
    }

    if (type === 'line') {
      return (
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#203754" />
            <XAxis dataKey="name" stroke="#9bb0c8" />
            <YAxis stroke="#9bb0c8" />
            <Tooltip />
            <Legend />
            {dataKeys.map((key, index) => (
              <Line key={key} type="monotone" dataKey={key} stroke={['#5ec3ff', '#4ade80', '#fbbf24', '#f87171'][index % 4]} strokeWidth={2.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#203754" />
          <XAxis dataKey="name" stroke="#9bb0c8" />
          <YAxis stroke="#9bb0c8" />
          <Tooltip />
          <Bar dataKey={dataKeys[0]} fill={color} radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="chart-card">
      <div className="section-heading">
        <h3>{title}</h3>
      </div>
      {renderChart()}
    </div>
  );
}

export default ChartCard;
