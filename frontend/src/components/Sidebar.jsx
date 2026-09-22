const navigationItems = [
  'Dashboard',
  'Video Analysis',
  'Detection Results',
  'Pollution Analysis',
  'Hotspots',
  'Environmental Report',
  'Research',
];

function Sidebar({ activePage, onSelect }) {
  return (
    <aside className="sidebar">
      <div className="brand-wrap">
        <div className="brand-mark">H</div>
        <div>
          <div className="brand-name">HydroWatch</div>
          <div className="brand-tag">Marine Intelligence</div>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="main navigation">
        {navigationItems.map((item) => (
          <button
            key={item}
            type="button"
            className={`nav-item ${activePage === item ? 'active' : ''}`}
            onClick={() => onSelect(item)}
          >
            {item}
          </button>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
