"""One module per panel.

Each module here draws exactly one region of the interface as plain functions
over a :class:`ui.widgets.Chrome`. Nothing in this package may import
``ui.ui_manager``: the manager owns the state and calls into the panels, never
the other way round.
"""
