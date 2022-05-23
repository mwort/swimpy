

# General

- all configuration parameters should be `dashboard.App` class attributes allowing per-project customisaiton in the `swimpy/settings.py` file


# Run tab

- run model button: `project.run`
- parameter subset:
    - this should be a subset of parameters defined in the Parameters tab, which should be set the same way
    - subset definition (parameter group, parameter name):
        - highlighted_parameters = [
            ("config_parameters", "iyr"),
            ("config_parameters", "nbyr"),
            ("basin_parameters", "sccor"),
            ("basin_parameters", "ecal"),
        ]
- the graph thats being updated when the model runs currently relies on reading the entire model output file every iteration via the `project.station_daily_discharge` property attribute. This is very slow and will have to be streamlined to only read the added lines. MW will provide support with this.


# Parameters tab

- parameters are organised in groups and have unique names, they are read and written via dict-like objects
- current parameter objects:
    - parameter_accessors = [
        "project.config_parameters",
        "project.basin_parameters",
    ]
- e.g. `project.basin_parameters.keys()` to read the names and `project.basin_parameters.update()` to write works just like dictionaries
- it's important to make the implementation flexible to be able to extent the parameter accessors by more than 2 objects, as other versions of the model have more.


# Output visualisation tabs

- the other tabs should be populated with model output visualisations that are configurable via function mappings.
- For example:
    output_tabs = {
        # Format: {"Tab title": [[row1-col1, row1-col2], [row2-col1, row2-col2, ...], ...]
            "Climate": [
                ["some.swimpy.graph_function", "some.swimpy.graph_function2"],
                ["some.swimpy.graph_function3", "some.swimpy.graph_function4"]
            ],
            "Maps": ...,
    }
- functions should be able to return:
    - HTML, e.g. a plotly plot or a table, which is embedded as is
    - matplotlib figures, these will be written to temporary png and embedded with <img> tags in the dashboard, this behaviour is already implemented in `modelmanager.plugins.browser.api.views.is_matplotlib_figure()`
    - `pandas.DataFrame`, these will converted to html tables via the `DataFrame.to_html()` method
    - example functions are:
        - plotly graph: `plotly_station_daily_discharge`
        - matplotlib figure: `project.catchment_annual_waterbalance.plot_mean`
        - pandas.DataFrame: `project.catchment_annual_waterbalance`
- Most output visualisation will take a little to render, so the tabs should be designed to first display the divs and then asynchronously call the functions