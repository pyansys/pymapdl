"""Plotting helper for MAPDL using pyvista"""
import pyvista as pv
import numpy as np

from ansys.mapdl.core.misc import unique_rows


def general_plotter(meshes, points, labels,
                    title='',
                    cpos=None,
                    show_bounds=False, show_axes=True,
                    background=None, off_screen=None,
                    screenshot=False,
                    window_size=None,
                    notebook=None,
                    # add_mesh kwargs:
                    color='w',
                    show_edges=None, edge_color=None, point_size=5.0,
                    line_width=None, opacity=1.0, flip_scalars=False,
                    lighting=None, n_colors=256,
                    interpolate_before_map=True, cmap=None,
                    render_points_as_spheres=False, render_lines_as_tubes=False,
                    stitle=None,
                    smooth_shading=None,
                    # labels kwargs
                    font_size=None,
                    font_family=None,
                    text_color=None):
    """General pyansys plotter for APDL geometry and meshes.

    Parameters
    ----------
    title : str, optional
        Add given title to plot.

    cpos : list(tuple(floats)), str
        The camera position to use.  You can either use a saved camera
        position or specify one of the following strings:

        - ``"xy"``
        - ``"xz"``
        - ``"yz"``
        - ``"yx"``
        - ``"zx"``
        - ``"zy"``
        - ``"iso"``

    off_screen : bool, optional
        Renders off screen when ``True``.  Useful for automated
        screenshots.

    window_size : list, optional
        Window size in pixels.  Defaults to ``[1024, 768]``

    notebook : bool, optional
        When True, the resulting plot is placed inline a jupyter
        notebook.  Assumes a jupyter console is active.  Automatically
        enables off_screen.

    show_bounds : bool, optional
        Shows mesh bounds when ``True``.

    show_axes : bool, optional
        Shows a vtk axes widget.  Enabled by default.

    screenshot : str or bool, optional
        Saves screenshot to file when enabled.

    color : string or 3 item list, optional, defaults to white
        Use to make the entire mesh have a single solid color.  Either
        a string, RGB list, or hex color string.  For example:
        ``color='white'``, ``color='w'``, ``color=[1, 1, 1]``, or
        ``color='#FFFFFF'``. Color will be overridden if scalars are
        specified.

    show_edges : bool, optional
        Shows the edges of a mesh.  Does not apply to a wireframe
        representation.

    edge_color : string or 3 item list, optional, defaults to black
        The solid color to give the edges when ``show_edges=True``.
        Either a string, RGB list, or hex color string.

    point_size : float, optional
        Point size of any nodes in the dataset plotted. Also applicable
        when style='points'. Default ``5.0``

    line_width : float, optional
        Thickness of lines.  Only valid for wireframe and surface
        representations.  Default None.

    opacity : float, str, array-like
        Opacity of the mesh. If a single float value is given, it will be
        the global opacity of the mesh and uniformly applied everywhere -
        should be between 0 and 1. A string can also be specified to map
        the scalars range to a predefined opacity transfer function
        (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        A string could also be used to map a scalars array from the mesh to
        the opacity (must have same number of elements as the
        ``scalars`` argument). Or you can pass a custom made transfer
        function that is an array either ``n_colors`` in length or shorter.

    n_colors : int, optional
        Number of colors to use when displaying scalars. Defaults to 256.
        The scalar bar will also have this many colors.

    cmap : str, list, optional
       Name of the Matplotlib colormap to us when mapping the ``scalars``.
       See available Matplotlib colormaps.  Only applicable for when
       displaying ``scalars``. Requires Matplotlib to be installed.
       ``colormap`` is also an accepted alias for this. If ``colorcet`` or
       ``cmocean`` are installed, their colormaps can be specified by name.

        You can also specify a list of colors to override an
        existing colormap with a custom one.  For example, to
        create a three color colormap you might specify
        ``['green', 'red', 'blue']``

    render_points_as_spheres : bool, optional
        Render points as spheres.

    render_lines_as_tubes : bool, optional
        Renders lines as tubes.

    smooth_shading : bool, optional
        Smoothly render curved surfaces when plotting.  Not helpful
        for all meshes.
    """
    if notebook:
        off_screen = True

    pl = pv.Plotter(off_screen=off_screen, notebook=notebook)

    if background:
        pl.set_background(background)

    for point in points:
        pl.add_points(point['points'],
                      scalars=point.get('scalars', None),
                      color=color,
                      show_edges=show_edges, edge_color=edge_color,
                      point_size=point_size, line_width=line_width,
                      opacity=opacity, flip_scalars=flip_scalars,
                      lighting=lighting, n_colors=n_colors,
                      interpolate_before_map=interpolate_before_map,
                      cmap=cmap, render_points_as_spheres=render_points_as_spheres,
                      render_lines_as_tubes=render_lines_as_tubes)

    for mesh in meshes:
        pl.add_mesh(mesh['mesh'],
                    scalars=mesh.get('scalars', None),
                    stitle=mesh.get('stitle', None),
                    color=mesh.get('color', color),
                    show_edges=show_edges, edge_color=edge_color,
                    smooth_shading=smooth_shading,
                    point_size=point_size, line_width=line_width,
                    opacity=opacity, flip_scalars=flip_scalars,
                    lighting=lighting, n_colors=n_colors,
                    interpolate_before_map=interpolate_before_map,
                    cmap=cmap, render_points_as_spheres=render_points_as_spheres,
                    render_lines_as_tubes=render_lines_as_tubes)

    for label in labels:
        # verify points are not duplicates
        points, idx, _ = unique_rows(np.array(label['points']))
        labels = np.array(label['labels'])[idx].tolist()

        pl.add_point_labels(points,
                            labels,
                            show_points=False, shadow=False,
                            font_size=font_size,
                            font_family=font_family,
                            text_color=text_color)

    if stitle is not None:
        pl.add_scalar_bar(title=stitle)

    if cpos:
        pl.camera_position = cpos

    if show_bounds:
        pl.show_bounds()

    if show_axes:
        pl.show_axes()
    if title:
        pl.add_title(title)
    if screenshot:
        pl.show(title=title, auto_close=False, window_size=window_size, screenshot=True)
        pl.screenshot(f'screenshot_{title}')
    else:
        pl.show()

    return pl.camera_position
