from dash import ALL, ctx
from src.layout import *
from matplotlib.colors import PowerNorm
from src.objects.Physics import *
from src.objects.Profile import *
from dash import no_update, Output, Input, State
import json

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

def plot_boundary_layer(_profile : Profile, _hypersonic : HypersonicObliqueShock):
    x = _profile.get_x()
    y = _profile.get_y()

    delta = _hypersonic.get_boundary_layer(x)


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

def plot_downstream(_profile, _hypersonic):
    mach_amb = _hypersonic.flow_characteristics['mach_amb']
    mach_amb[np.abs(mach_amb) > _hypersonic.mach_inf] = 0
    mach_amb[mach_amb < 0] = 0

    x = _profile.get_x()

    nb_plot = 2 * 3 # Based on the subplot from Pierre
    y_labels = ["Pression [Pa]", "Temperature [K]", "Densité [kg⋅m⁻³]", "Mach", "Vitesse du son [m⋅s⁻¹]", "Vitesse [m⋅s⁻¹]"]
    labels = ["Pression", "Temperature", "Densité", "Mₙ", "aₙ", "Vₙ"]
    dic_val = ["pressure", "temperature", "density", "mach_n", "soundspeed_n", "velocity_n"]

    figures = []
    for y_label, var, label in zip(y_labels, dic_val, labels):
        figure = go.Figure(layout=dark_graph_layout)
        figure.update_layout(
            title=f"Evolution de l'écoulement caractéristique après impact \n à mach {_hypersonic.mach_inf} à"
                  f"z = {_hypersonic.physic.atm.altitude} mètres")
        figure.add_trace(
            go.Scatter(
                x=x,
                y=_hypersonic.flow_characteristics[var],
                line=dict(
                    color='navy'
                ),
                name=label
            )
        )

        if label == "Pression":
            figure.update_yaxes(type="log")
        elif var == "mach_n":
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=mach_amb,
                    line=dict(
                        color='#7AB2D3',
                        dash='dash'
                    ),
                    name="Mach ambiant"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=np.full(len(x), _hypersonic.mach_inf),
                    line=dict(
                        color='#3B1E54',
                        dash='dot',
                        width=2
                    ),
                    name=r"$$Mach_{infini}$$"
                )
            )
        elif var == "velocity_n":
            v_amb = _hypersonic.flow_characteristics[var] * mach_amb
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=v_amb,
                    line=dict(
                        color='#7AB2D3',
                        dash='dashdot',
                    ),
                    name="Norme de la vitesse"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=np.full(len(x), _hypersonic.physic.velocity_x),
                    line=dict(
                        color='#3B1E54',
                        dash='dot',
                        width=2
                    ),
                    name="Vitesse d'écoulement libre"
                )
            )
        elif var == "soundspeed_n":
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=np.full(len(x), _hypersonic.sound_speed),
                    line=dict(
                        color='#3B1E54',
                        dash='dot',
                        width=2
                    ),
                    name="Vitesse du son en écoulement libre"
                )
            )

        figure.update_yaxes(title_text=y_label)

        figures.append(figure)

    return figures

def plot_contour(_profile : Profile, _hypersonic : HypersonicObliqueShock):
    x = _profile.get_x()
    y = _profile.get_y()

    x_shock_curve = _hypersonic.x_shock_curve
    y_shock_curve = _hypersonic.y_shock_curve

    x_extension = np.linspace(-10, 0, 1000)
    y_extension = np.zeros_like(x_extension)

    x_contour = np.concatenate([x_extension, x])
    y_contour = np.concatenate([y_extension, y])

    x_mesh = x_contour
    y_mesh = np.linspace(-np.max(y_shock_curve) - 0.5, np.max(y_shock_curve) + 0.5, len(y_contour))
    x_grid, y_grid = np.meshgrid(x_mesh, y_mesh)

    # split shock curve
    index_before_0 = np.where(x_shock_curve < 0)[0]
    x_shock_curve_neg = x_shock_curve[index_before_0[0]:index_before_0[-1] + 1]
    x_shock_curve_pos = x_shock_curve[index_before_0[-1] + 1:]

    index_in_x_extension = np.array([np.abs(x_extension - x_vals).argmin() for x_vals in x_shock_curve_neg])
    mapping_before_0 = {ext_index: shock_index for ext_index, shock_index in zip(index_in_x_extension, index_before_0)}

    # matrix definition
    pressure_matrix = np.full((len(y_mesh), len(x_mesh)), _hypersonic.physic.atm.pressure)
    temperature_matrix = np.full((len(y_mesh), len(x_mesh)), _hypersonic.physic.atm.temperature)
    density_matrix = np.full((len(y_mesh), len(x_mesh)), _hypersonic.physic.atm.density)

    for index in range(len(y)):
        if np.min(x_mesh) <= x[index] <= np.max(x_mesh):
            upper_bound = y[index]
            lower_bound = -y[index]

            x_index = np.argmin(np.abs(x_mesh - x[index]))
            x_index_shock = np.argmin(np.abs(x_shock_curve - x[index]))

            inside_mask = (y_grid[:, x_index] >= lower_bound) & (y_grid[:, x_index] <= upper_bound)
            between_profile_shock_mask_upper = (y_grid[:, x_index] <= y_shock_curve[x_index_shock]) & (y_grid[:, x_index] > y[index])
            between_profile_shock_mask_lower = (y_grid[:, x_index] >= -y_shock_curve[x_index_shock]) & (y_grid[:, x_index] < -y[index])

            # pressure
            pressure_matrix[inside_mask, x_index] = np.nan
            pressure_matrix[between_profile_shock_mask_upper, x_index] = _hypersonic.flow_characteristics['pressure'][index]
            pressure_matrix[between_profile_shock_mask_lower, x_index] = _hypersonic.flow_characteristics['pressure'][index]

            # temperature
            temperature_matrix[inside_mask, x_index] = np.nan
            temperature_matrix[between_profile_shock_mask_upper, x_index] = _hypersonic.flow_characteristics['temperature'][index]
            temperature_matrix[between_profile_shock_mask_lower, x_index] = _hypersonic.flow_characteristics['temperature'][index]

            # density
            density_matrix[inside_mask, x_index] = np.nan
            density_matrix[between_profile_shock_mask_upper, x_index] = _hypersonic.flow_characteristics['density'][index]
            density_matrix[between_profile_shock_mask_lower, x_index] = _hypersonic.flow_characteristics['density'][index]


    if len(_profile.get_section().keys()) > 1:
        x_curve_index = np.where((x_shock_curve <= x[-1]) & (x_shock_curve >= x[-1] - _hypersonic.stand_off_distance_arr[-1]) & (x >= x[-1]))[0]
        x_profile_index = np.where((x <= x[-1]) & (x >= x[-1] - _hypersonic.stand_off_distance_arr[-1]))[0]
        interpolated_y_curve = np.linspace(y_shock_curve[x_curve_index][0], y_shock_curve[x_curve_index][-1], len(x_profile_index))

        for index in range(len(x_profile_index)):
            upper_bound = interpolated_y_curve[index]
            lower_bound = y[x_profile_index[index]]

            shock_curve_upper_mask = (y_grid[:, x_profile_index[index]] <= upper_bound) & (y_grid[:, x_profile_index[index]] >= lower_bound)
            shock_curve_lower_mask = (y_grid[:, x_profile_index[index]] >= -upper_bound) & (y_grid[:, x_profile_index[index]] <= -lower_bound)

            # pressure
            pressure_matrix[shock_curve_upper_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['pressure'])
            pressure_matrix[shock_curve_lower_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['pressure'])

            # temperature
            temperature_matrix[shock_curve_upper_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['temperature'])
            temperature_matrix[shock_curve_lower_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['temperature'])

            # density
            density_matrix[shock_curve_upper_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['density'])
            density_matrix[shock_curve_lower_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['density'])


    for index in range(len(y_extension)):
        if index in index_in_x_extension:
            corresponding_index = mapping_before_0[index]
            upper_bound = y_shock_curve[corresponding_index]
            lower_bound = -y_shock_curve[corresponding_index]

            between_shock_mask = (y_grid[:, index] >= lower_bound) & (y_grid[:, index] <= upper_bound)

            # pressure
            pressure_matrix[between_shock_mask, index] = np.max(_hypersonic.flow_characteristics['pressure'])

            # temperature
            temperature_matrix[between_shock_mask, index] = np.max(_hypersonic.flow_characteristics['temperature'])

            # density
            density_matrix[between_shock_mask, index] = np.max(_hypersonic.flow_characteristics['density'])

    figures = []

    contour_label = ["Variation de pression [Pa]", "Variation de température [K]", "Variation de densité [kg.m^{-3}]"]
    matrix_variable = [pressure_matrix, temperature_matrix, density_matrix]
    key_var = ["pressure", "temperature", "density"]

    for matrix_var, c_label, key in zip(matrix_variable, contour_label, key_var):
        norm = PowerNorm(gamma=0.4, vmin=np.nanmin(matrix_var), vmax=np.nanmax(matrix_var))
        matrix_normed = norm(matrix_var)

        figure = go.Figure(
            go.Contour(
                z=matrix_normed,
                x=np.linspace(-1, 1, matrix_normed.shape[1]),
                y=np.linspace(-1, 1, matrix_normed.shape[0]),
                colorscale='Jet',
                contours=dict(start=0, end=1, size=0.01),
                colorbar=dict(title="Variable")
            ),
            layout=dark_graph_layout
        )

        figure.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black'), name='Profil'))
        figure.add_trace(go.Scatter(x=x, y=-y, mode='lines', line=dict(color='black'), showlegend=False))

        figure.add_trace(
            go.Scatter(x=x, y=0.8 * y, mode='lines', line=dict(color='#F5F5F7', dash='dash', width=0.5),
                       name='Shock Layer'))
        figure.add_trace(
            go.Scatter(x=x, y=-0.8 * y, mode='lines', line=dict(color='#F5F5F7', dash='dash', width=0.5),
                       showlegend=False))

        figure.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y, -y[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,0,0.5)',
            line=dict(color='black'),
            name='Profil Area',
        ))

        figure.update_layout(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            legend=dict(x=0.02, y=0.98),
        )

        figures.append(figure)

    return figures

def define_callbacks3(app):
    @app.callback(
        Output({'type': 'graph-grid', 'index': ALL}, 'figure'),
        Input('ok-button-calcul', 'n_clicks'),
        State('profile-store', 'data'),
        State('physics-store', 'data'),
        prevent_initial_call=True
    )
    def calcul(_n_clicks, _profile_dict, _physics_dict):
        profile: Profile = load_profile_from_dict(json.loads(_profile_dict))
        physics = Physics()
        physics.from_dict(json.loads(_physics_dict))

        hypersonic = HypersonicObliqueShock(_physic=physics, _profile=profile)

        # plot the profile
        figure = plot_the_shock_along_profile(profile, hypersonic)

        # Deviation angle
        figure_deviation = plot_deviation_angle(profile, hypersonic)

        # boundary layer
        figure_boundary = plot_boundary_layer(profile, hypersonic)

        # downstream graphic
        figure_downstream_pressure, figure_downstream_temperature, figure_downstream_density,\
            figure_downstream_mach, figure_downstream_soundspeed, figure_downstream_velocity = plot_downstream(profile, hypersonic)

        figure_contour_pressure, figure_contour_temperature, figure_contour_density = plot_contour(profile, hypersonic)

        print('return')
        return figure, figure_deviation, figure_boundary, figure_downstream_pressure, figure_downstream_temperature, \
            figure_downstream_density, figure_downstream_mach, figure_downstream_soundspeed, figure_downstream_velocity, \
            figure_contour_pressure, go.Figure(layout=dark_graph_layout), go.Figure(layout=dark_graph_layout)


    @app.callback(
        Output('highlight-store', 'data'),
        Input({'type': 'grid-item', 'index': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def highlight_graphs(_n_clicks):
        if ctx.triggered:
            clicked_index = ctx.triggered[0]['prop_id'].split('.')[0].split('index":')[1].split(',')[0]
            return clicked_index



