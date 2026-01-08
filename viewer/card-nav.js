(() => {
  const pages = [
    { title: 'Home', description: 'Landing hub and intro.', href: 'index.html', accent: '#e23d2d' },
    { title: 'Main viewer', description: 'Explore station data and charts.', href: 'viewer.html', accent: '#0f8c86' },
    { title: 'Compare', description: 'Side-by-side station plots.', href: 'compare.html', accent: '#0f1818' }
  ];

  const getCurrentFile = () => {
    const path = (window.location.pathname || '').toLowerCase().replace(/\\/g, '/');
    const bits = path.split('/').filter(Boolean);
    return bits.pop() || 'index.html';
  };

  const staggerDelay = (index) => `${index * 60}ms`;

  const buildCards = (grid) => {
    const current = getCurrentFile();
    grid.innerHTML = '';
    pages.forEach((page, idx) => {
      const card = document.createElement('a');
      card.className = 'rb-cardnav__card';
      card.href = page.href;
      card.style.setProperty('--rb-card-accent', page.accent);
      card.style.transitionDelay = staggerDelay(idx);
      const hrefFile = (page.href || '').split('/').filter(Boolean).pop()?.toLowerCase();
      if (hrefFile && hrefFile === current) {
        card.classList.add('is-active');
        card.setAttribute('aria-current', 'page');
      }

      card.innerHTML = `
        <div class="rb-cardnav__title">${page.title}</div>
        <div class="rb-cardnav__desc">${page.description}</div>
      `;

      grid.appendChild(card);
    });
  };

  const setup = (container) => {
    const toggle = container.querySelector('.rb-cardnav__toggle');
    const panel = container.querySelector('[data-rb-panel]');
    const inner = container.querySelector('[data-rb-inner]');
    const grid = container.querySelector('[data-rb-grid]');
    if (!panel || !inner || !grid) return;

    buildCards(grid);

    container.classList.add('is-open');
    panel.style.height = 'auto';
    panel.setAttribute('aria-hidden', 'false');

    if (!toggle) return;

    toggle.setAttribute('aria-expanded', 'true');
    toggle.setAttribute('aria-label', 'Close menu');
    toggle.addEventListener('click', () => {
      const isOpen = container.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(isOpen));
      toggle.setAttribute('aria-label', isOpen ? 'Close menu' : 'Open menu');
      panel.style.height = isOpen ? 'auto' : '0px';
      panel.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
    });
  };

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-rb-cardnav]').forEach(setup);
  });
})();
