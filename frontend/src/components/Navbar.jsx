function Navbar({ title, subtitle }) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">HydroWatch</p>
        <h1>{title}</h1>
      </div>
      <div className="topbar-summary">
        <span className="status-dot" />
        <span>{subtitle}</span>
      </div>
    </header>
  );
}

export default Navbar;
