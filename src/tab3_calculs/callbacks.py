from src.layout import *
import numpy as np
import json


def define_callbacks3(app):
    @app.callback(
        Output('results-graphs', 'figure'),
        Input('ok-button-calcul', 'n_clicks'),
        State('profile-store', 'data'),
        prevent_initial_call=True
    )
    def calcul(_n_clicks, _profile):
        _profile = json.loads(_profile)
        if _profile is {}:
            return

        if _profile['name'] == 'Conical':
            profile = Profile.Conical()
        elif _profile['name'] == 'Parabolic':
            profile = Profile.Parabolic()
        else:
            return

        profile.from_dict(_profile)

        x = profile.get_x()
        y = profile.get_y()
        panel_num = len(x) - 1
        panels = np.array([x[:-1], x[1:], y[:-1], y[1:]]).T

        dx = panels[:, 1] - panels[:, 0]
        dy = panels[:, 3] - panels[:, 2]
        lengths = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)

        figure = go.Figure(layout=dark_graph_layout)
        for i in range(panel_num):
            figure.add_trace(go.Scatter(x=[panels[i, 0], panels[i, 1]], y=[panels[i, 2], panels[i, 3]]))

        return figure