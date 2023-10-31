import marimo

__generated_with = "0.1.38"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        """
        ##### Copyright 2019 Google LLC.

        SPDX-License-Identifier: Apache-2.0
        """
    )
    return


@app.cell
def _():
    #@title License
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    return


@app.cell
def _(mo):
    mo.md(
        """
        *If there's an option to "OPEN IN PLAYGROUND" at top left, click that so you can run through this notebook yourself.*

        #Introduction
        ---

        The goal of this [Colab](https://colab.sandbox.google.com/) notebook is to highlight some benefits of using Google BigQuery and Colab together to perform some common data science tasks. We will go through:
        * using Python data science tools to do some analysis/curve fitting
        * creating some interactive outputs
        * using Python functionality "on top" of BigQuery to scale analysis
        * writing some analysis results back into BigQuery

        ### Data Set and Analysis Goals
        The data set we'll be working with involves global weather data, specifically daily temperature readings from around the world. These are found in the [BigQuery public dataset "noaa_gsod"](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=noaa_gsod&page=dataset), provided by the [National Oceanic and Atmospheric Administration](https://www.noaa.gov/).

        Our initial goal is to do some exploratory analysis of daily temperature data: getting data for a single place (interactively, so we can pick a few different places one-by-one) for 2019, then multiple years, plotting it, etc. From there, we motivate the idea of fitting a curve to the daily temperature data to capture temperature patterns in a given location with useful summary statistics (e.g. range in average temperature across the year). Then, we apply that curve fitting procedure to several weather locations and examine further some of the more interesting results.

        ### Related Data Science Tools/Techniques
        In each step of the above journey, we show how BigQuery and Python data science and plotting libraries (via Colab) can work together to enable this type of analysis at scale. Some key tools and techniques we'll employ:
        *   [Google BigQuery](https://cloud.google.com/bigquery/what-is-bigquery)
        *   Python Data Science Libraries: [pandas](https://pandas.pydata.org/), [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)
        *   [Plotly Python](https://plot.ly/python/) for visualization
        *   [Sinusoidal model](https://en.wikipedia.org/wiki/Sinusoidal_model) and [curve fitting](https://en.wikipedia.org/wiki/Curve_fitting)
        """
    )
    return

app._unparsable_cell(
    r"""
    #@title Install Latest Version of Some Packages
    !pip install --upgrade google-cloud-bigquery
    !pip install --upgrade google-cloud-bigquery-storage
    !pip install --upgrade pyarrow
    !pip install --upgrade google-cloud-core
    !pip install --upgrade chart_studio
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        """
        ***Check the output from the cell above, as it may require restarting the Colab runtime for the upgrades to take effect in the environment. You can pick up from this point after restarting.***
        """
    )
    return


app._unparsable_cell(
    r"""
    #@title Import Python Libraries & Some Other Setup
    # Basic Python data science libraries
    import pandas as pd
    import numpy as np
    import scipy.optimize

    # Import and setup for plotly in Colab
    import chart_studio
    import chart_studio.plotly as py
    import plotly.graph_objects as go
    import plotly.io as pio

    # Enable extension package to display pandas data frames as interactive tables
    %load_ext google.colab.data_table
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        """
        The authentication step in the next cell will require manually going through some pop-up screens and copy/pasting an authentication code from another window back into the cell to complete (on the 1st run; may run automatically thereafter).
        """
    )
    return


@app.cell
def _():
    #@title Provide Google Credentials to Colab Runtime (May Require Manually Copy/Pasting Authentication Code)
    from google.colab import auth
    auth.authenticate_user()
    print('Authenticated')
    return auth,


@app.cell
def _(mo):
    mo.md(
        """

        ##Please enter your own GCP/BigQuery project ID in the form below, then run the cell to set up the BigQuery client.

        If you don’t already have a GCP project, there are [2 free options available](https://cloud.google.com/bigquery/):

        1. For BigQuery specifically, sign up for [BigQuery sandbox](https://cloud.google.com/blog/products/data-analytics/query-without-a-credit-card-introducing-bigquery-sandbox) (1 TB query, 10 GB storage capacity per month).
        2. If you want to experiment with multiple GCP products, activate the [free trial](https://cloud.google.com/free/) ($300 credit for up to 12 months).
        """
    )
    return


@app.cell
def _():
    #@title Enter GCP/BigQuery Project ID
    project_id = 'gcp-data-science-demo' #@param{type:"string"}

    # Packages used for interfacing w/ BigQuery from Python
    from google.cloud import bigquery
    from google.cloud import bigquery_storage_v1beta1

    # Create BigQuery client
    bq_client = bigquery.Client(project = project_id)

    # Create BigQuery storage client
    bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()
    return (
        bigquery,
        bigquery_storage_v1beta1,
        bq_client,
        bq_storage_client,
        project_id,
    )


@app.cell
def _(mo):
    mo.md(
        """
        #Examine Daily Temperature Data for a Single Weather Station
        ----
        In this section, we look at how to get daily temperature data for a single weather station - initially, for 2019 only, then for multiple years together.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        Our first task is to see which weather stations are available and have "reasonably good" data. We rely on the bigquery "magic" '%%bigquery' and pass in the project_id as a parameter up top, storing the results in a pandas data frame called "weather_stations".

        The query below pulls those from the 'stations' table, but with some filtering to look only at stations that have:
        * a "beginning date" January 1, 2000 or prior
        * an "end date" through at least June 30, 2019
        * at least 95% of possible 2019 dates with a "valid" temperature

        The first 2 restrictions are implemented with a simple filter in the WHERE clause. The last one is bit more complicated and takes up more of the query, using a WITH clause to find the # of 2019 dates by station and the max # of possible dates in 2019, then using an INNER JOIN and CROSS JOIN on the way to additional filtering for the 95% criteria above. Doing this high-level "data validation" up front ensures that we only consider weather stations with reasonably complete recent temperature data as we proceed to do more with that later on.
        """
    )
    return


app._unparsable_cell(
    r"""
    #@title Get Weather Stations w/ Mostly Complete 2019 Daily Temp from BigQuery
    %%bigquery weather_stations --project {project_id}

    # Subquery to count # of dates w/ valid temperature data by station
    WITH
    Num2019TempDatesByStation AS
    (
      SELECT
        daily_weather.stn,

        # Count # of distinct dates w/ temperature data for each station
        COUNT(DISTINCT
          # Convert year/month/day info into date
          DATE(
            CAST(daily_weather.year AS INT64),
            CAST(daily_weather.mo AS INT64),
            CAST(daily_weather.da AS INT64)
            )) AS num_2019_temp_dates

      FROM
        `bigquery-public-data.noaa_gsod.gsod2019` daily_weather

      WHERE
        daily_weather.temp IS NOT NULL AND
        daily_weather.max IS NOT NULL AND
        daily_weather.min IS NOT NULL AND
        # Remove days w/ missing temps coded as 99999.9
        daily_weather.temp != 9999.9 AND
        daily_weather.max != 9999.9 AND
        daily_weather.min != 9999.9

      GROUP BY
        daily_weather.stn
    ),

    # Calculate max number of 2019 temperature dates across all stations
    MaxNum2019TempDates AS
    (
      SELECT
        MAX(num_2019_temp_dates) AS max_num_2019_temp_dates

      FROM
        Num2019TempDatesByStation
    )

    SELECT
      Stations.*,
      Num2019TempDatesByStation.num_2019_temp_dates

    FROM
      `bigquery-public-data.noaa_gsod.stations` Stations

    # Inner join to filter to only stations present in 2019 data
    INNER JOIN
      Num2019TempDatesByStation ON (
        stations.usaf = Num2019TempDatesByStation.stn
        )

    # Cross join to get max number on each row, to use in filtering below
    CROSS JOIN
      MaxNum2019TempDates

    WHERE
      # Filter to stations that have had tracking since at least 1/1/2000
      Stations.begin <= '20000101' AND
      # Filter to stations that have had tracking through at least 6/30/2019
      Stations.end >= '20190630' AND
      # Filter to stations w/ >= 90% of the max number of dates for 2019
      Num2019TempDatesByStation.num_2019_temp_dates >=
        (0.90 * MaxNum2019TempDates.max_num_2019_temp_dates)

    ORDER BY
      stations.usaf
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        """
        Once this query runs, we can look at "weather_stations" in an interactive table. This is one advantage of running this query in Colab as opposed to the BigQuery terminal - we can immediately sort and filter the output in various different ways without any exporting or running additional queries. Having this output displayed in the Colab allows us to reference it when selecting stations by USAF in other parts of this Colab later on.
        """
    )
    return


@app.cell
def _(weather_stations):
    #@title Interactive Table of Weather Stations
    weather_stations
    return


@app.cell
def _(mo):
    mo.md(
        """
        Find a weather station of interest from the table above, and enter its USAF number into the form below. Forms are another feature of Colab that allow you to do interactive exploratory analysis. Our original default USAF is 745090, representing [Moffett Federal Airfield](https://en.wikipedia.org/wiki/Moffett_Federal_Airfield), which is very close to Google headquarters. A couple interesting USAFs to try: 242660, 825910, 890090, 974060.

        You can see below that we also set the chosen station USAF as a [BigQuery parameter](https://cloud.google.com/bigquery/docs/parameterized-queries), to be used in the next step.
        """
    )
    return


@app.cell
def _(weather_stations):
    #@title Choose Weather Station by USAF (If Not in Above Table, Random One Is Chosen)
    chosen_station_usaf = "745090" #@param{type:"string"}

    if chosen_station_usaf not in weather_stations['usaf'].tolist():
      print('Not a Valid USAF, Picking Random Weather Station Instead...')
      chosen_station_usaf = weather_stations['usaf'].sample(1).iloc[0]

    # Filter to only chosen station
    chosen_station_info = weather_stations[weather_stations['usaf'] ==
      chosen_station_usaf]

    chosen_station_name = chosen_station_info['name'].iloc[0]

    # Add station usaf to BigQuery parameters dictionary
    bigquery_params = {
      "chosen_station_usaf": chosen_station_usaf
      }

    print('Chosen Station: ' + chosen_station_name)
    chosen_station_info
    return (
        bigquery_params,
        chosen_station_info,
        chosen_station_name,
        chosen_station_usaf,
    )


@app.cell
def _(mo):
    mo.md(
        """
        Below we use BigQuery to get 2019 daily temperature data (avg, min, max) for the station chosen above. We pass on the station USAF in the bigquery params argument on the line with the %%bigquery magic, and it's used in the WHERE clause ("@chosen_station_usaf") to filter the query to only the station of interest.
        """
    )
    return


app._unparsable_cell(
    r"""
    #@title Get Daily Temperature Data for Chosen Station (Single Year)
    %%bigquery chosen_station_daily_2019 --project {project_id} --params $bigquery_params

    SELECT
      # Station information
      daily_weather.stn AS usaf,

      # Convert year/month/day info into date
      DATE(
        CAST(daily_weather.year AS INT64),
        CAST(daily_weather.mo AS INT64),
        CAST(daily_weather.da AS INT64)
        ) AS date,

      daily_weather.temp AS avg_temp,
      daily_weather.count_temp AS n_for_avg_temp,

      daily_weather.max AS max_temp,
      daily_weather.flag_max AS max_temp_flag,

      daily_weather.min AS min_temp,
      daily_weather.flag_min AS min_temp_flag

    FROM
      `bigquery-public-data.noaa_gsod.gsod2019` daily_weather

    WHERE
      # Filter to only chosen station
      daily_weather.stn = @chosen_station_usaf AND
      # Remove days w/ missing temps coded as 99999.9 (can throw off calculations)
      daily_weather.temp != 9999.9 AND
      daily_weather.max != 9999.9 AND
      daily_weather.min != 9999.9

    ORDER BY
      date DESC
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(
        """
        Once the query runs, we call the resulting data frame to get an interactive table of results.
        """
    )
    return


@app.cell
def _(chosen_station_daily_2019):
    #@title Interactive Table of Daily Temperature Data for Chosen Station
    chosen_station_daily_2019
    return


@app.cell
def _(mo):
    mo.md(
        """
        It would be easier to "see" trends in a plot of temperature over time than in a table. In the next cell, we use plotly to generate an interactive time series plot of the average, max, and min temperature at our chosen weather station on each day of 2019. We wrap the plotting code in a function which takes in the data frame and fields to be plotted, so that we can re-use that code later.

        Depending on which station you choose, you may see lots of changing temperature over time (up or down), wild swings from day to day, or something fairly consistent throughout the year. The setup here allows picking a few different stations in the form a couple cells above, one at a time, and running through this section of code to examine the 2019 plot for each station.
        """
    )
    return


@app.cell
def _(chosen_station_daily_2019, chosen_station_name, go, pd, pio):
    #@title Plot of Daily Temperature Data for Chosen Station (Single Year)

    # Create table of temperature series to plot, with names, symbols, colors
    daily_temp_plot_fields = pd.DataFrame.from_records(
      columns = ['field_name', 'plot_label', 'marker_symbol', 'line_color',
        'plot_mode'],
      data = [
        ('avg_temp', 'Avg', 'circle', None, 'markers'),
        ('max_temp', 'Max', 'triangle-up', None, 'markers'),
        ('min_temp', 'Min', 'triangle-down', None, 'markers')
        ]
      )

    # Create function to plot single station daily temperature
    def plot_single_station_daily_temp(daily_temp_data, plot_fields, station_name):
      daily_plot_data = []

      for index, row in plot_fields.iterrows():
        daily_plot_data = (daily_plot_data +
          [go.Scatter(
            x = daily_temp_data['date'],
            y = daily_temp_data[row['field_name']],
            name = row['plot_label'],
            marker = dict(
              # Constant color scale for plotting temp to use for all stations
              cmin = -22, # -22°F corresponds to -30°C (very cold, to most)
              cmax = 122, # 122°F corresponds to 50°C (very hot, to most)
              color = daily_temp_data[row['field_name']],
              # colorscale = 'BlueReds',
              colorscale = [[0, 'rgb(0, 0, 230)'], [0.5, 'rgb(190, 190, 190)'],
                [1, 'rgb(230, 0, 0)']],
              symbol = row['marker_symbol']
              ),
            line = dict(
              color = row['line_color']
              ),
            mode = row['plot_mode']
            )]
          )

      daily_plot_layout = go.Layout(
        title = dict(
          text = (station_name + ' Daily Temperature'),
          xref = "paper",
          x = 0.5
          ),
        yaxis = dict(title = 'Temperature (°F)')
        )

      pio.show(go.Figure(daily_plot_data, daily_plot_layout))

    plot_single_station_daily_temp(chosen_station_daily_2019,
      daily_temp_plot_fields, chosen_station_name)
    return daily_temp_plot_fields, plot_single_station_daily_temp


@app.cell
def _(mo):
    mo.md(
        """
        The above plot may be informative for 2019, but we don't have enough data to see seasonal patterns or other larger trends over longer periods of time. To get this, we'll have to go back into our [BigQuery NOAA data set](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=noaa_gsod&page=dataset). This has data back to 1929, but not in one single table - rather, each year has its own table named "gsod{year}" in that dataset. To get multiple years together in BigQuery, we can "UNION ALL" a set of similar subqueries, each pointing to the table for a single year, e.g.:

        ( SELECT * FROM `bigquery-public-data.noaa_gsod.gsod2005` WHERE stn = '745090')

        UNION ALL

        ( SELECT * FROM `bigquery-public-data.noaa_gsod.gsod2006` WHERE stn = '745090')

        UNION ALL

        [...]

        UNION ALL

        ( SELECT * FROM `bigquery-public-data.noaa_gsod.gsod2019` WHERE stn = '745090')


        This is repetitive to do "by hand" and annoying to have to recreate if we end up changing our date range, for example.

        This is another place where Python can help us. We can use the pattern of the subquery text - each yearly subquery is the same except for the last 4 characters on the table name - to loop over our years of interest and create the SQL text we need to get the data for multiple years together. The function in the cell below creates this multi-year daily weather SQL statement, then executes a query that is very similar to the one for 2019 above, but instead of querying the 'gsod2019' table only, we use the multi-year table in the FROM clause to get all the same information across multiple years.

        Different starting and ending years between 1929 and 2019 can be chosen in the cell below (original defaults were 2005 and 2019), then run to see the interactive scatterplot of temperature at the chosen weather station for the desired span.
        """
    )
    return


@app.cell
def _(
    bq_client,
    bq_storage_client,
    chosen_station_name,
    chosen_station_usaf,
    daily_temp_plot_fields,
    np,
    plot_single_station_daily_temp,
):
    #@title Get and Plot Multi-Year Daily Temperature Data for Chosen Station
    chosen_start_year = 2005 #@param{type:"integer"}

    chosen_end_year = 2019 #@param{type:"integer"}

    def get_single_station_daily_temp_multiple_yrs(station_usaf, start_year,
      end_year):

      single_station_daily_weather_multiyear_union_sql = ("\nUNION ALL\n".
        join([('''
          ( SELECT * FROM `bigquery-public-data.noaa_gsod.gsod{year}`
          WHERE stn = '{station_usaf}')
          ''')
        .format(year = year, station_usaf = station_usaf)
           for year in np.arange(start_year, (end_year + 1))
        ]))

      single_station_daily_multiyear_sql = '''
        WITH
        daily_weather AS
        (
          {daily_weather_table}
        )

        SELECT
          daily_weather.stn AS usaf,

          # Convert year/month/day info into date
          DATE(
            CAST(daily_weather.year AS INT64),
            CAST(daily_weather.mo AS INT64),
            CAST(daily_weather.da AS INT64)
            ) AS date,

          daily_weather.temp AS avg_temp,
          daily_weather.count_temp AS n_for_avg_temp,

          daily_weather.max AS max_temp,
          daily_weather.flag_max AS max_temp_flag,

          daily_weather.min AS min_temp,
          daily_weather.flag_min AS min_temp_flag

        FROM
          daily_weather

        WHERE
          # Remove days w/ missing temps coded as 99999.9 (can throw off calcs)
          daily_weather.temp != 9999.9 AND
          daily_weather.max != 9999.9 AND
          daily_weather.min != 9999.9

        ORDER BY
          date DESC
        '''

      single_station_daily_multiyear_query = (single_station_daily_multiyear_sql.
        format(
          daily_weather_table = single_station_daily_weather_multiyear_union_sql,
          station_usaf = station_usaf
          )
        )

      single_station_daily_multiyear_df = (bq_client.
        query(single_station_daily_multiyear_query).
        result().
        to_arrow(bqstorage_client = bq_storage_client).
        to_pandas()
        )

      return(single_station_daily_multiyear_df)

    chosen_station_daily_multiyear = get_single_station_daily_temp_multiple_yrs(
      chosen_station_usaf, chosen_start_year, chosen_end_year
      )

    plot_single_station_daily_temp(chosen_station_daily_multiyear,
      daily_temp_plot_fields, chosen_station_name)
    return (
        chosen_end_year,
        chosen_start_year,
        chosen_station_daily_multiyear,
        get_single_station_daily_temp_multiple_yrs,
    )


@app.cell
def _(mo):
    mo.md(
        """
        There is a wide range in global temperature and how much it moves across time, so the plot above can take on many different shapes depending on the location chosen. The plotting code is set to "fit" the y-axis dynamically according to the data on the plot, so "swings" of the same height across different station plots don't usually represent the same temperature difference. The constant color scale does help put into perspective warmer vs colder places, though.

        Despite the variation in weather, most locations will show some regular temperature pattern, cycling relatively smoothly up and down, repeating each year. This makes sense because we have [seasons on Earth](https://spaceplace.nasa.gov/seasons/en/)!

        Again, the setup here allows picking a few different stations in the form a couple cells above, one at a time, and running through this section of code to examine the temperature trends across multiple years for each station.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Fit Sine Curve to Single Weather Station Temperature Data
        ----
        So far, we've used BigQuery and Colab together to do some exploratory analysis of temperature over time for a couple locations across the world. We've probably found that some places are hot, others are cold, and a lot move between hot and cold throughout the year every year - with some being more extreme than others.

        Given the seasonal pattern we've seen in most cases, one logical next step would be to do some curve fitting to get a summary of a location's temperature movements over the course of multiple years. This would allow us to get a smoothed estimate of average temperature - better than directly averaging, especially if we don't have complete data (e.g. some readings are missing) or have partial years of data (e.g. missing end of 2019) - as well as an estimate of the annual range in temperature (high - low, across the year) at the given location. This is another task where a language like Python with solid optimization/modeling libraries can help us supplement what we are doing with BigQuery.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        An applicable curve to fit here is a [sine wave](https://en.wikipedia.org/wiki/Sine_wave), a mathematical function that describes a smooth periodic oscillation - like temperature moving in consistent patterns across days over years. We use [scipy's curve fit optimization method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit) to estimate a [sinusoidal model](https://en.wikipedia.org/wiki/Sinusoidal_model) for the daily temperature at a given weather station. The functions are set up to run the optimization to fit 4 parameters - mean, amp (for amplitude), freq (frequency), and phase_shift - and return either those parameters or the estimated daily temperature values from the model for that station. [This Stack Overflow post](https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy) provides some more useful technical details on how to fit this type of curve using these Python libraries.
        """
    )
    return


@app.cell
def _(np, pd, scipy):
    #@title Functions to Fit Sine Curve to Daily Multi-Year Temperature Data
    # Describe sinusoidal model as function w/ parameters of interest
    def sine_function(t, amp, freq, phase_shift, mean):
      return (amp * np.sin(freq * 2 * np.pi * (t - phase_shift)) + mean)

    # Fit sinusoidal model to data, return either fit info or daily temp estimates
    def fit_sine_curve_to_daily_temp_data(daily_temp_data, temp_field_name,
      return_value = 'sine curve fit info'):

      # Calculate total range of days in data
      daily_temp_data['days_since_start'] = (daily_temp_data['date'] -
        min(daily_temp_data['date'])).dt.days

      # Starting point for mean is mean of temp in data set
      guess_mean = daily_temp_data[temp_field_name].mean()

      # Starting point for amplitude is half diff btw 1st & 99th %tiles of temp
      guess_amp = (daily_temp_data[temp_field_name].quantile(0.99) -
        daily_temp_data[temp_field_name].quantile(0.01)) / 2

      # Starting point for frequency is inverse of avg # of days in year
      guess_freq = 1/365.25

      # Starting point for phase shift is +80 days (into spring, in most cases)
      guess_phase_shift = 80

      # Use curve fit optimizer on data, w/ above guesses as starting points
      sine_curve_fit = scipy.optimize.curve_fit(
        f = sine_function,
        xdata = np.array(daily_temp_data['days_since_start']),
        ydata = np.array(daily_temp_data[temp_field_name]),
        p0 = [guess_amp, guess_freq, guess_phase_shift, guess_mean]
        )

      # Extract estimated parameters from curve fit
      est_amp, est_freq, est_phase_shift, est_mean = sine_curve_fit[0]

      # Use sine function & parameters to get daily estimates of average temperature
      daily_temp_data['est_' + temp_field_name] = sine_function(
        daily_temp_data['days_since_start'],
        est_amp, est_freq, est_phase_shift, est_mean
        )

      # Calculate mean absolute error of estimates vs actual temperature
      curve_estimate_mean_abs_err = abs(
        daily_temp_data['est_' + temp_field_name] - daily_temp_data[temp_field_name]
        ).mean()

      # Create data frame of sine curve fit info
      sine_curve_fit_info_df = pd.DataFrame(data = [{
        ('est_amp_' + temp_field_name): est_amp,
        ('est_freq_' + temp_field_name): est_freq,
        ('est_phase_shift_' + temp_field_name): est_phase_shift,
        ('est_mean_' + temp_field_name): est_mean,
        ('est_range_' + temp_field_name): 2 * abs(est_amp),
        ('mae_fitted_' + temp_field_name): curve_estimate_mean_abs_err
        }])

      # Return either sine curve fit into or daily temp data w/ estimates
      if(return_value == 'sine curve fit info'):
        return(sine_curve_fit_info_df)

      elif(return_value == 'daily temp data with estimates'):
        return(daily_temp_data)
    return fit_sine_curve_to_daily_temp_data, sine_function


@app.cell
def _(mo):
    mo.md(
        """
        In the next cell, we actually fit the sinusoidal model to the average temperature data from our chosen weather station, then add the estimates from our model to the daily temperature plot for the given station. *(Going forward, we only look at the "avg" temperature field, though we could look at max/min using the same setup.)*
        """
    )
    return


@app.cell
def _(
    chosen_station_daily_multiyear,
    chosen_station_name,
    fit_sine_curve_to_daily_temp_data,
    pd,
    plot_single_station_daily_temp,
):
    #@title Add Estimated Avg Temp and Plot Alongside Actual Temp for Selected Weather Station
    # Use function to fit sine curve, get out daily temp estimates for given station
    chosen_station_daily_temp_with_preds = fit_sine_curve_to_daily_temp_data(
      daily_temp_data = chosen_station_daily_multiyear,
      temp_field_name = 'avg_temp',
      return_value = 'daily temp data with estimates'
      )

    # Set up plot fields structure: points for actual temp, curve for estimated temp
    daily_avg_and_estimate_plot_fields = pd.DataFrame.from_records(
      columns = ['field_name', 'plot_label', 'marker_symbol', 'line_color',
        'plot_mode'],
      data = [
        ('avg_temp', 'Actual Avg', 'circle', None, 'markers'),
        ('est_avg_temp', 'Estimated Avg', None, 'purple', 'lines')
        ]
      )

    # Use function to plot daily temperature with estimates for given station
    plot_single_station_daily_temp(chosen_station_daily_temp_with_preds,
      daily_avg_and_estimate_plot_fields, chosen_station_name)
    return (
        chosen_station_daily_temp_with_preds,
        daily_avg_and_estimate_plot_fields,
    )


@app.cell
def _(mo):
    mo.md(
        """
        In the case of most weather stations, the plot shows that the sinusoidal model ends up being a fairly good fit for the average temperature trend across years. It's a good way to capture the natural variation in the data, while limiting the impact of outliers on specific days in specific years, so that we can get solid estimates of average temperature and the range between high and low (average) temperature throughout the year.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Fit Individual Sine Curves and Extract Temperature-Related Attributes for Multiple Weather Stations
        ---
        After running through the sine curve fits for a couple different stations in the section above, we might see that the attributes of the fit (mean, amplitude, etc.) provide useful summary information about the longer-term temperature trends at that location. If we wanted to go a step further to see which places have similar temperature attributes, or find ones at the most extreme (hottest/coldest, most/least varying over the year), it would make sense to fit the model and store results for *multiple* weather stations together, instead of just a few run one at a time.

        One way to do this would be to rewrite our daily temperature query above to get data from *all* weather stations of interest, read that into Colab, and repeat the procedure by station. The problem with that approach is that we have more than 6000 weather stations and are looking across roughly 5300 time points per station - more than 30 million rows of data, much more than we'd typically want to read into Python. Instead, we can loop over stations, getting the data from BigQuery, fitting the sinusoidal model, extracting and storing off the summary stats one station at a time. This is another way we can put BigQuery and Python together to let each tool do some pieces that it is individually good for, then combine the results.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        Use the forms in the cell below to specify 2 things:

        1.   The number of stations to sample randomly. Can go from a couple all the way up to the total number of stations [above](https://colab.sandbox.google.com/drive/1W_RPEBgItNuwp4QexlfWwjuXEdLOpVX8#scrollTo=hR-NNHjPisSs), keeping in mind that the procedure takes longer for more stations (up to a few hours if you choose all of them). Original default is 100.
        2.   An array of USAFs (in string form) for other stations that you want to include in the analysis (not random, but selectively chosen). The original default represents a set of USAFs that are extreme in one way or another: ['242660', '825910', '890090', '974060'].

        The code will then loop through the complete set of (random + selectively chosen) stations, collecting attributes of each individual curve fit into a data frame, which is then combined and output into an interactive table at the end.
        """
    )
    return


@app.cell
def _(
    chosen_end_year,
    chosen_start_year,
    fit_sine_curve_to_daily_temp_data,
    get_single_station_daily_temp_multiple_yrs,
    pd,
    weather_stations,
):
    #@title Choose Some # of Random Weather Stations & Other Specific Ones to Include in Analysis
    # Choose number of weather stations to sample (randomly) from above list
    num_stations_to_sample = 100 #@param {type:"number"}

    # Enter USAF #s of other weather stations to be included (quoted & separated by commas)
    other_usafs_to_include = "['242660', '825910', '890090', '974060']" #@param {type:"string"}

    # Seed for random # generation to ensure consistent sampling (reproducibility)
    seed = 23

    chosen_weather_stations = pd.concat([
      # Randomly sample specified number of weather stations
      weather_stations.sample(n = num_stations_to_sample, random_state = seed),
      # Filter to other specified stations provided in array of USAFs
      weather_stations.query("usaf in " + other_usafs_to_include)
      ],
      ignore_index = True
      # Might be duplicates if sampled & fixed stations overlap, so drop them
      ).drop_duplicates()

    # Initialize list of sine curve fit info data frames
    sine_curve_fit_info_df_collection = []

    # Loop over data frame of chosen weather stations
    for index, row in chosen_weather_stations.iterrows():
      # Use function to get daily temperature data for given station from BigQuery
      this_station_daily_temp_data = get_single_station_daily_temp_multiple_yrs(
        station_usaf = row['usaf'],
        start_year = chosen_start_year,
        end_year = chosen_end_year
        )

      # Don't count unless station has >=500 days of temperature data
      if(this_station_daily_temp_data.shape[0] < 500):
        # Print message and move on in this case
        print("Not Enough Temp Data for USAF " + row['usaf'] + ' ' + row['name'])

      # As long as station has >=500 days of temperature data
      else:
        # Use function to find sine curve fit for this station's temperature data
        this_station_temp_sine_curve_fit_info = fit_sine_curve_to_daily_temp_data(
          daily_temp_data = this_station_daily_temp_data,
          temp_field_name = 'avg_temp'
          )

        # Add station USAF and name to this fit into data frame
        this_station_temp_sine_curve_fit_info['station_usaf'] = row['usaf']
        this_station_temp_sine_curve_fit_info['station_name'] = row['name']

        # Add data frame for this station to collection for all stations
        sine_curve_fit_info_df_collection = (sine_curve_fit_info_df_collection +
          [this_station_temp_sine_curve_fit_info])

    # Concatenate collection of all stations' data frames into 1 data frame
    all_station_fit_info = pd.concat(sine_curve_fit_info_df_collection,
      ignore_index = True).set_index(['station_usaf', 'station_name']).reset_index()

    # Look at interactive table of all station fit info
    all_station_fit_info
    return (
        all_station_fit_info,
        chosen_weather_stations,
        index,
        num_stations_to_sample,
        other_usafs_to_include,
        row,
        seed,
        sine_curve_fit_info_df_collection,
        this_station_daily_temp_data,
        this_station_temp_sine_curve_fit_info,
    )


@app.cell
def _(mo):
    mo.md(
        """
        Sorting and filtering the table above allows us to find interesting weather stations - hottest, coldest, lowest variation, and more.

        The cell below replicates code from above to generate a sinusoidal model fit and plot for a given weather station (by USAF chosen in the form) - it is put here so we can conveniently pick off a station from the table above and study its daily temperature plot below (and repeat relatively quickly with other stations).

        For example, you may want to pick USAF 242660, representing [Verhojansk, Russia](https://en.wikipedia.org/wiki/Verkhoyansk), a town near the Arctic Circle that has an estimate range of average temperature over 110 degrees(!), among the highest in our data set. That shows up as some serious amplitude on the sine curve on the plot!
        """
    )
    return


@app.cell
def _(
    all_station_fit_info,
    chosen_end_year,
    chosen_start_year,
    chosen_weather_stations,
    daily_avg_and_estimate_plot_fields,
    fit_sine_curve_to_daily_temp_data,
    get_single_station_daily_temp_multiple_yrs,
    plot_single_station_daily_temp,
):
    #@title Look at Sine Curve Fit Stats & Plot for Single Weather Station in Chosen Set
    station_usaf = '242660' #@param{type:"string"}

    # Message if station is not in our chosen set
    if station_usaf not in chosen_weather_stations['usaf'].tolist():
      print('Not in Chosen Weather Stations')

    # Message if station was in our chosen set, but not enough temperature data
    elif station_usaf not in all_station_fit_info['station_usaf'].tolist():
      print('Not Enough Temp Data for USAF ' + station_usaf)

    else:
      # Filter to only chosen station
      station_fit_info = all_station_fit_info[
        all_station_fit_info['station_usaf'] == station_usaf]

      # Print fit into
      print(station_fit_info.round(decimals = 4))

      # Extract weather station name
      station_name = station_fit_info['station_name'].iloc[0]

      # Use function to get daily temperature data for given station from BigQuery
      station_daily_temp_data = get_single_station_daily_temp_multiple_yrs(
        station_usaf = station_usaf,
        start_year = chosen_start_year,
        end_year = chosen_end_year
        )

      # Use function to find sine curve fit for this station's temperature data
      station_daily_temp_data_with_preds = fit_sine_curve_to_daily_temp_data(
        daily_temp_data = station_daily_temp_data,
        temp_field_name = 'avg_temp',
        return_value = 'daily temp data with estimates'
        )

      # Use function to plot given station's daily temperature with model estimates
      plot_single_station_daily_temp(station_daily_temp_data_with_preds,
        daily_avg_and_estimate_plot_fields, station_name)
    return (
        station_daily_temp_data,
        station_daily_temp_data_with_preds,
        station_fit_info,
        station_name,
        station_usaf,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Write All Weather Stations Data Back to BigQuery
        ---
        A final step in this Colab/Python and BigQuery journey is to take some of what we created here and put it back into BigQuery. The station summary statistics we got in the curve fitting might be useful to have for other analyses, whether those be looking at more weather data or using the weather summary to join with another data set.

        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        To take a Pandas data frame and load it into BigQuery, we use the [BigQuery client function "load_table_from_dataframe"](https://google-cloud.readthedocs.io/en/latest/bigquery/generated/google.cloud.bigquery.client.Client.load_table_from_dataframe.html), with appropriate output dataset and table info. Another option is [pandas' own function for uploading to BigQuery](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_gbq.html), but the BigQuery client library is the official way to do this ([more info on migration here](https://cloud.google.com/bigquery/docs/pandas-gbq-migration)).

        The forms in the cell below allow you to specify the output dataset and table IDs, as well as if you'd like to replace or append the results to the BigQuery table.
        """
    )
    return


@app.cell
def _(all_station_fit_info, bigquery, bq_client, pd):
    #@title Write All Weather Station Data to BigQuery Table
    output_dataset_id = 'weather_demo' #@param{type:'string'}

    output_table_id = 'sample_weather_station_temp_curve_fit_info' #@param{type:'string'}

    replace_or_append_output = 'replace' #@param{type:'string'} ['replace', 'append']

    # Combine project and dataset
    project_dataset = (bq_client.project + '.' + output_dataset_id)

    # Check to make sure output dataset exists, create it if not
    try:
     bq_client.get_dataset(output_dataset_id)
     print("Dataset " + project_dataset + " exists\n")

    except:
     print("Dataset " + project_dataset + " doesn't exist, so creating it\n")
     dataset = bq_client.create_dataset(bigquery.Dataset(project_dataset))

    job_config = bigquery.LoadJobConfig()

    # Modify job config depending on if we want to replace or append to table
    if(replace_or_append_output == 'replace'):
     job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    else:
     job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

    dataset_ref = bq_client.dataset(output_dataset_id)
    table_ref = dataset_ref.table(output_table_id)

    # Get timestamp (UTC), add to data frame at granularity of seconds
    all_station_fit_info['timestamp'] = pd.Timestamp.now(tz = 'UTC').ceil(freq = 's'
     )

    # Use client functionality to load BigQuery table from Pandas data frame
    bq_client.load_table_from_dataframe(
     dataframe = all_station_fit_info,
     destination = table_ref,
     job_config = job_config
     ).result()

    print('All Station Fit Info output (' + replace_or_append_output + ') to ' +
     project_dataset + '.' + output_table_id +
     '\n')
    return (
        dataset,
        dataset_ref,
        job_config,
        output_dataset_id,
        output_table_id,
        project_dataset,
        replace_or_append_output,
        table_ref,
    )


@app.cell
def _(mo):
    mo.md(
        """
        Once the cell above has run successfully, you should able to see your weather station temperature summary outputs back in BigQuery...ready for use in your next data analysis!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

