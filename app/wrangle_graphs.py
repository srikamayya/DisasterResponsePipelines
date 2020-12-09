import pandas as pd
import plotly.graph_objs as go
import joblib

def return_graphs(df):
    """Creates four plotly visualizations
    Args:
        None
    Returns:
        list (dict): list containing the four plotly visualizations
    """

    # extract data for visuals
    categories = df.iloc[:,4:40].sum().sort_values(ascending=False)
    categories_names = categories.index.to_series().apply(lambda x: x.replace('_', ' '))
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #scatter = joblib.load("../data/scatter_data.pkl")

    # creating the visuals
    graph_one = []
    graph_one.append(
      go.Bar(
      x = categories_names,
      y = categories
      )
    )
    layout_one = dict(title = 'Distribution of Message Categories',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'Count'),
                )
    
    graph_two = []
    graph_two.append(
      go.Bar(
      x = genre_names,
      y = genre_counts,
      )
    )
    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre'),
                yaxis = dict(title = 'Count'),
                )

    # append all charts to the figures list
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))

    return graphs