from dash import no_update, ALL, ctx

from src.layout import *
import numpy as np
import json


from src.objects.Physics import *
from src.objects.Profile import *

def plot_the_shock_along_profile(_profile, _hypersonic):
    x = _profile.get_x()
    y = _profile.get_y()

    radius_arr = [section['radius'] for section in _profile.get_section().values() if 'radius' in section]
    y_min, y_max = -5 * radius_arr[-1], 5 * radius_arr[-1]

    figure = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            name='Profil',
            fill='tozeroy',
            fillpattern=dict(shape="/"),
            line=dict(color='grey')),
        layout=dark_graph_layout)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=-y,
            name='Profil (Symétrie)',
            fill='tozeroy',
            fillpattern=dict(shape="/"),
            line=dict(color='grey')
        ))
    figure.update_layout(title=_profile.to_dict()['class'])

    # shock layer
    figure.add_trace(
        go.Scatter(
            x=_hypersonic.x_shock_curve,
            y=_hypersonic.y_shock_curve,
            name="Couche de choc",
            line=dict(color='red',
                      dash='longdash')
        )
    )
    figure.add_trace(
        go.Scatter(
            x=_hypersonic.x_shock_curve,
            y=-_hypersonic.y_shock_curve,
            name="Couche de choc (Symétrie)",
            line=dict(
                color='red',
                dash='longdash'
            )
        )
    )

    for index, (section_name, section_value) in enumerate(_profile.get_section().items()):
        if index == 0:
            figure.add_trace(
                go.Scatter(
                    x=[x[index], x[index]],
                    y=[y_min, y_max],
                    line=dict(
                        color='#FF6500',
                        dash='dot'
                    ),
                    name=f"Section {section_name}",
                    opacity=0.5
                )
            )

        else:
            figure.add_trace(
                go.Scatter(
                    x=[section_value['x'][0], section_value['x'][0]],
                    y=[y_min, y_max],
                    line=dict(
                        color='#FF6500',
                        dash='dot'
                    ),
                    name=f"Section {section_name}",
                    opacity=0.5
                )
            )

    figure.update_xaxes(title_text="Mètres")
    figure.update_yaxes(title_text="Mètres")

    return figure


def plot_deviation_angle(_profile, _hypersonic):
    x = _profile.get_x()

    mu = np.arcsin(np.divide(1, _hypersonic.mach_inf))

    figure = go.Figure(
        data=go.Scatter(
            x=x,
            y=np.degrees(_hypersonic.theta),
            line=dict(
                color='navy'
            ),
            name=f"Angle de déviation theta"
        ),
        layout=dark_graph_layout
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=np.degrees(_hypersonic.beta),
            line=dict(
                color='red',
                dash='dash'
            ),
            name="Angle de choc beta"
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=np.full(len(x), mu),
            line=dict(
                color='purple',
                dash='dashdot'
            ),
            name="Angle de mach mu"
        )
    )

    figure.update_xaxes(title_text="Unité de longueur")
    figure.update_yaxes(title_text="Angle (Degrés)")

    return figure

def boundary_layer_figure(_profile : Profile, _hypersonic : HypersonicObliqueShock):
    x = _profile.get_x()
    y = _profile.get_y()

    """if re_local > 3000:
        r = 0.90"""
    delta = _hypersonic.get_boundary_layer(x)

    print(delta)
    print(np.nanmax(delta))

    figure = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            name='Profil',
            fill='tozeroy',
            fillpattern=dict(shape="/"),
            line=dict(color='grey')),
        layout=dark_graph_layout)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=-y,
            name='Profil (Symétrie)',
            fill='tozeroy',
            fillpattern=dict(shape="/"),
            line=dict(color='grey')
        ))
    figure.update_layout(title=_profile.to_dict()['class'])

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y + delta
        )
    )

    return figure


def define_callbacks3(app):
    @app.callback(
        Output({'type': 'graph-grid', 'index': 0}, 'figure'),
        Output({'type': 'graph-grid', 'index': 1}, 'figure'),
        Output({'type': 'graph-grid', 'index': 2}, 'figure'),
        Input('ok-button-calcul', 'n_clicks'),
        State('profile-store', 'data'),
        State('physics-store', 'data'),
        prevent_initial_call=True
    )
    def calcul(_n_clicks, _profile_dict, _physics_dict):
        profile: Profile.Profile = load_profile_from_dict(json.loads(_profile_dict))
        physics = Physics()
        physics.from_dict(json.loads(_physics_dict))


        hypersonic = HypersonicObliqueShock(_physic=physics, _profile=profile)

        # plot the profile
        figure = plot_the_shock_along_profile(profile, hypersonic)

        # Deviation angle
        figure_deviation = plot_deviation_angle(profile, hypersonic)

        # boundary layer
        figure_boundary = boundary_layer_figure(profile, hypersonic)

        return figure, figure_deviation, figure_boundary


    @app.callback(
        Output('highlight-store', 'data'),
        Input({'type': 'grid-item', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def highlight_graphs(_n_clicks):
        if ctx.triggered:
            clicked_index = ctx.triggered[0]['prop_id'].split('.')[0].split('index":')[1].split(',')[0]
            return clicked_index



