import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def add_galactic(ra, dec):
    c = SkyCoord(ra=np.asarray(ra) * u.degree, dec=np.asarray(dec) * u.degree, frame='icrs')
    return c.galactic.l.deg, c.galactic.b.deg


def mollweide_scatter(ax, lon, lat, groups, styles, title):
    lon_rad = np.radians(lon)
    lon_rad = np.where(lon_rad > np.pi, lon_rad - 2 * np.pi, lon_rad)
    lat_rad = np.radians(lat)
    for name in sorted(styles):
        mask = np.array(groups) == name
        if mask.sum() == 0:
            continue
        sty = styles[name]
        ax.scatter(-lon_rad[mask], lat_rad[mask],
                   s=3, label=name,
                   c=sty['color'], marker=sty['marker'])
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.3)
    ax.legend(loc='lower right', fontsize=7, markerscale=2)
