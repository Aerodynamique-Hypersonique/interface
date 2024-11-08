from dash import ALL, ctx
from src.layout import *
from matplotlib.colors import PowerNorm
from src.objects.Physics import *
from src.objects.Profile import *
from dash import Output, Input, State, no_update
import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import datetime

figure_path = 'assets/plots_images/'

def plot_the_shock_along_profile(_hypersonic):
    x = _hypersonic.profile.get_x()
    y = _hypersonic.profile.get_y()

    radius_arr = [section['radius'] for section in _hypersonic.profile.get_section().values() if 'radius' in section]
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
    figure.update_layout(title="Onde de choc autour du profil")

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


    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex='all')

    axs[0].plot(x, y, color='grey', label='profil')
    axs[0].plot(x, -y, color='grey', linestyle='-.', label='symetry')
    axs[0].fill_between(x, y, -y, color='grey', hatch='//', alpha=0.5)

    axs[0].plot(_hypersonic.x_shock_curve, _hypersonic.y_shock_curve, color='red', linestyle='--', label='shock layer')
    axs[0].plot(_hypersonic.x_shock_curve, -_hypersonic.y_shock_curve, color='red', linestyle='--')

    for index, (section_name, section_value) in enumerate(_hypersonic.profile.get_section().items()):
        if index == 0:
            axs[0].vlines(section_value['x'][index], y_min, y_max, color='#FF6500', linestyle=':',
                          alpha=0.75)
            figure.add_trace(
                go.Scatter(
                    x=[section_value['x'][index], section_value['x'][index]],
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
            axs[0].vlines(section_value['x'][index - 1], y_min, y_max, color='#FF6500', linestyle=':',
                          alpha=0.75)
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

    axs[0].grid('on')
    axs[0].set_ylabel(r"$Profil$")
    axs[0].legend(loc='upper right')

    axs[0].set_title('Studying Profil')

    axs[1].plot(x, np.degrees(_hypersonic.theta), color='navy', label=r"Angle de déviation $\theta$")
    axs[1].plot(x, np.degrees(_hypersonic.beta), color='red', linestyle='--', label=r"Angle de choc $\beta$")
    axs[1].plot(x, np.full(len(x), _hypersonic.physic.atm.mu), color='purple', linestyle='-.', label=r"Angle de Mach $\mu$")
    axs[1].legend(loc='upper right')
    axs[1].grid('on')
    axs[1].set_xlabel(r"$Unit\ of\ length$")
    axs[1].set_ylabel(r"$Angle\ [\degree]$")
    axs[1].set_ylim([np.rad2deg(np.nanmin(_hypersonic.beta[np.isfinite(_hypersonic.beta)])) - 20,
                     np.rad2deg(np.nanmax(_hypersonic.theta[np.isfinite(_hypersonic.theta)])) + 20])

    plt.savefig(figure_path + 'Geometry.png', dpi=300, transparent=True, bbox_inches="tight")

    figure.update_xaxes(title_text="Mètres")
    figure.update_yaxes(title_text="Mètres")


    return figure

def plot_deviation_angle(_hypersonic):
    x = _hypersonic.profile.get_x()

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

    figure.update_layout(title="Angle de déviation")

    figure.update_xaxes(title_text="Unité de longueur")
    figure.update_yaxes(title_text="Angle (Degrés)")

    return figure

def plot_boundary_layer(_hypersonic : HypersonicObliqueShock):
    x = _hypersonic.profile.get_x()
    y = _hypersonic.profile.get_y()

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
    figure.update_layout(title=_hypersonic.profile.to_dict()['class'])

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y + delta
        )
    )

    figure.update_layout(title="Couche Limite")

    return figure

def plot_downstream(_hypersonic):
    mach_amb = _hypersonic.flow_characteristics['mach_amb']
    mach_amb[np.abs(mach_amb) > _hypersonic.mach_inf] = 0
    mach_amb[mach_amb < 0] = 0

    x = _hypersonic.profile.get_x()

    fig, axs = plt.subplots(3, 2, figsize=(15, 9), sharex='all')
    fig.suptitle(f'Evolution of flow characteristics after impact\nat Mach {_hypersonic.mach_inf:.10f}, z = {_hypersonic.physic.atm.altitude} m',
                 fontsize=16)

    y_labels = ["Pression [Pa]", "Temperature [K]", "Densité [kg⋅m⁻³]", "Mach", "Vitesse du son [m⋅s⁻¹]", "Vitesse [m⋅s⁻¹]"]
    labels = ["Pression", "Temperature", "Densité", "Mₙ", "aₙ", "Vₙ"]
    dic_val = ["pressure", "temperature", "density", "mach_n", "soundspeed_n", "velocity_n"]

    figures = []
    for ax, y_label, var, label in zip(axs.flat, y_labels, dic_val, labels):
        ax.plot(x, _hypersonic.flow_characteristics[var], color='navy', label=label)

        figure = go.Figure(layout=dark_graph_layout)
        figure.update_layout(
            title=f"Evolution de l'écoulement caractéristique après impact")
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
            ax.set_yscale('log')
            figure.update_yaxes(type="log")
        elif var == "mach_n":
            ax.plot(x, mach_amb, color="#7AB2D3", linestyle='-.', label='Mach ambiant')
            ax.plot(x, np.full(len(x), _hypersonic.mach_inf), color="#3B1E54", linestyle=':', linewidth=2,
                    label=r'$Mach_{\infty}$')

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
                    name="Machₐₗₓ"
                )
            )
        elif var == "velocity_n":
            v_amb = _hypersonic.flow_characteristics[var] * mach_amb

            ax.plot(x, v_amb, color="#7AB2D3", linestyle='-.', label='Velocity norm')
            ax.plot(x, np.full(len(x), _hypersonic.physic.velocity_x), color="#3B1E54", linestyle=':', linewidth=2,
                    label='Free Flow Velocity')


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
            ax.plot(x, np.full(len(x), _hypersonic.sound_speed), color="#3B1E54", linestyle=':', linewidth=2,
                    label='Free Flow Sound Speed')

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

        ax.set_ylabel(y_label)
        ax.grid('on')
        ax.legend(loc='upper right')

        figure.update_yaxes(title_text=y_label)

        figures.append(figure)

    plt.savefig(figure_path + 'DownstreamVariables.png', dpi=300, transparent=True, bbox_inches="tight")

    return figures

def plot_contour(_hypersonic : HypersonicObliqueShock):
    x = _hypersonic.profile.get_x()
    y = _hypersonic.profile.get_y()

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


    if len(_hypersonic.profile.get_section().keys()) > 1:
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
            temperature_matrix[shock_curve_lower_mask, x_profile_index[index] + len(x_extension)] = np.max(_hypersonic.flow_characteristics['temperature'][:4])

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

    paths = []

    contour_label = ["Variation de pression [Pa]", "Variation de température [K]", "Variation de densité [kg.m^{-3}]"]
    matrix_variable = [pressure_matrix, temperature_matrix, density_matrix]
    key_var = ["pressure", "temperature", "density"]

    for matrix_var, c_label, key in zip(matrix_variable, contour_label, key_var):
        norm = PowerNorm(gamma=0.4, vmin=np.nanmin(matrix_var), vmax=np.nanmax(matrix_var))

        fig, ax = plt.subplots(figsize=(7, 6))

        c_variable = ax.contourf(x_grid, y_grid, matrix_var, levels=125, cmap='jet', norm=norm)

        fig.colorbar(c_variable, ax=ax, label=c_label)
        ax.plot(x, y, color='black', label='Profil')
        ax.plot(x, -y, color='black')
        ax.plot(_hypersonic.x_shock_curve, _hypersonic.y_shock_curve, color='#F5F5F7', linestyle='--',
                           linewidth=0.5, label='Couche de choc')
        ax.plot(_hypersonic.x_shock_curve, -_hypersonic.y_shock_curve, color='#F5F5F7', linestyle='--',
                           linewidth=0.5)
        ax.fill_between(x, -y, y, color='black', alpha=0.5, hatch='//', label='Profil')
        ax.legend(loc='upper left')

        # Save the individual subplot
        timestamp = datetime.datetime.now().timestamp()  # Used to resolve the client page's cache and force it to load a new image
        fig.savefig(f"{figure_path}ContourGraphic_{key}_{timestamp}.png", dpi=300, transparent=True, bbox_inches="tight")



        paths.append(f"{figure_path}ContourGraphic_{key}_{timestamp}.png")
        plt.close(fig)
        print('Saved ContourGraphic for key:', key)

        print('ok7')

    return paths


def define_callbacks3(app):
    @app.callback(
        Output('calcul-store', 'data'),
        Input('ok-button-calcul', 'n_clicks'),
        State('profile-store', 'data'),
        State('physics-store', 'data'),
        prevent_initial_call=True
    )
    def calcul(_n_clicks, _profile_dict, _physics_dict):
        path_to_remove = 'assets/plots_images/*'
        r = glob.glob(path_to_remove)
        for i in r:
            os.remove(i)

        if _profile_dict is not None and _profile_dict != {} and _physics_dict is not None and _physics_dict != {}:
            profile: Profile = load_profile_from_dict(json.loads(_profile_dict))
            physics = Physics()
            physics.from_dict(json.loads(_physics_dict))

            hypersonic = HypersonicObliqueShock(_physic=physics, _profile=profile)
            hypersonic.calcul()

            return hypersonic.to_json()
        return no_update


    @app.callback(
        Output({'type': 'graph-grid', 'index': 0}, 'figure'),
        Output({'type': 'graph-grid', 'index': 1}, 'figure'),
        Output({'type': 'graph-grid', 'index': 2}, 'figure'),
        Input('calcul-store', 'data'),
        prevent_initial_call=True
    )
    def plot_first_3_graphs(_hypersonic_data):
        hypersonic = HypersonicObliqueShock()
        hypersonic.from_dict(json.loads(_hypersonic_data))

        figure_shock = plot_the_shock_along_profile(hypersonic)
        figure_shock.update_layout(showlegend=False)

        # Deviation angle
        figure_deviation = plot_deviation_angle(hypersonic)
        figure_deviation.update_layout(showlegend=False)

        # boundary layer
        figure_boundary = plot_boundary_layer(hypersonic)
        figure_boundary.update_layout(showlegend=False)


        return figure_shock, figure_deviation, figure_boundary

    @app.callback(
        Output({'type': 'graph-grid', 'index': 3}, 'figure'),
        Output({'type': 'graph-grid', 'index': 4}, 'figure'),
        Output({'type': 'graph-grid', 'index': 5}, 'figure'),
        Output({'type': 'graph-grid', 'index': 6}, 'figure'),
        Output({'type': 'graph-grid', 'index': 7}, 'figure'),
        Output({'type': 'graph-grid', 'index': 8}, 'figure'),
        Input('calcul-store', 'data'),
        prevent_initial_call=True
    )
    def plot_second_6_graphs(_hypersonic_data):
        hypersonic = HypersonicObliqueShock()
        hypersonic.from_dict(json.loads(_hypersonic_data))

        # downstream graphic
        [figure_downstream_pressure, figure_downstream_temperature, figure_downstream_density,
            figure_downstream_mach, figure_downstream_soundspeed, figure_downstream_velocity] = plot_downstream(hypersonic)

        figure_downstream_pressure.update_layout(showlegend=False)
        figure_downstream_temperature.update_layout(showlegend=False)
        figure_downstream_density.update_layout(showlegend=False)
        figure_downstream_mach.update_layout(showlegend=False)
        figure_downstream_soundspeed.update_layout(showlegend=False)
        figure_downstream_velocity.update_layout(showlegend=False)

        return figure_downstream_pressure, figure_downstream_temperature, figure_downstream_density, \
            figure_downstream_mach, figure_downstream_soundspeed, figure_downstream_velocity

    @app.callback(
        Output({'type': 'image', 'index': 9}, 'src'),
        Output({'type': 'image', 'index': 10}, 'src'),
        Output({'type': 'image', 'index': 11}, 'src'),
        Input('calcul-store', 'data'),
        prevent_initial_call=True
    )
    def plot_last_3_graphs(_hypersonic_data):
        hypersonic = HypersonicObliqueShock()
        hypersonic.from_dict(json.loads(_hypersonic_data))
        paths = plot_contour(hypersonic)



        """figure_contour_pressure.update_layout(showlegend=False)
        figure_contour_temperature.update_layout(showlegend=False)
        figure_contour_density.update_layout(showlegend=False)"""

        return paths[0], paths[1], paths[2]

    @app.callback(
        Output('highlighted-graph', 'children', allow_duplicate=True),
        Output('highlighted-graph', 'style', allow_duplicate=True),
        Input({'type': 'grid-item', 'index': ALL}, 'n_clicks'),
        State({'type': 'graph-grid', 'index': ALL}, 'figure'),
        prevent_initial_call=True
    )
    def highlight_graph(_div_clicked, _figures):
        button_id = ctx.triggered_id.index
        figure_dict = _figures[button_id]

        figure = go.Figure(figure_dict)
        figure.update_layout(showlegend=True)

        return dcc.Graph(figure=figure, style=dict(height='70vh')), \
                dict(visiblity='visible')

    @app.callback(
        Output('highlighted-graph', 'children', allow_duplicate=True),
        Output('highlighted-graph', 'style', allow_duplicate=True),
        Input({'type': 'grid-item-img', 'index': ALL}, 'n_clicks'),
        State({'type': 'image', 'index': ALL}, 'src'),
        prevent_initial_call=True
    )
    def highlight_img(_div_clicked, _images_path):
        button_id = ctx.triggered_id.index
        image_path = _images_path[button_id - 9]

        return html.Img(style={'backgroundColor': '#2c2f33', 'max-height': '40vh', 'width': '100vw', 'object-fit': 'contain'}, src=image_path), \
            dict(visiblity='visible', height='40vh')

