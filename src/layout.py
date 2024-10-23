from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
from src.objects import Physics, Profile

dark_graph_layout = go.Layout(
    paper_bgcolor='#2c2f33',  # Dark background outside the plotting area
    plot_bgcolor='#1d1f21',   # Darker background inside the plotting area
    font=dict(color='#f0f0f0'),  # White font for axis labels
    xaxis=dict(showgrid=False, zeroline=False),  # Remove grids and zero lines
    yaxis=dict(showgrid=False, zeroline=False),
    margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins for better spacing
)

def get_layout():
    return html.Div([
    dcc.Store(id='profile-store'),
    dcc.Store(id='physics-store'),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Création du profil', value='tab-1', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
                 selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            html.H1(children='Création du profil'),
                html.Div(children=[
                    html.Div(children=[
                        dcc.Dropdown(['Parabolique', 'Conique', 'Ogive'], id='dropdown-shape'),
                        html.Button("OK", id='ok-button-profile', className='ok-button', n_clicks=0),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '40%', 'margin': '0 auto', 'gap': '15px'}),

                    html.Div(children=[
                        html.B("Taille du profil", id='text-input-length'),
                        dcc.Input(id='input-length', placeholder='Entrez la taille du profil', type='number'),
                        html.B("Angle du profil", id='text-input-angle'),
                        dcc.Input(id='input-angle', placeholder="Entrez l'angle du profil", type='number'),
                        html.Div(children=[], id='div-dynamic-components'),
                    ], id='div-selection-attributes'),
                ], id='div-selection'),

                dcc.Graph(id='shape-graphs', figure=go.Figure(layout=dark_graph_layout), style={'height': '100vh'})
        ]),

        # Second Tab: Valeurs initiales
        dcc.Tab(label='Valeurs initiales', value='tab-2', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
            selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            dcc.Checklist(options=[{'label': 'Rendre les valeurs modifiables', 'value': 'modify'}], id='modify-values-check',
                          style={'float': 'right', 'color': '#f0f0f0'}),
            html.H1(children='Valeurs initiales', style={'color': '#f0f0f0', 'flex-grow': '1', 'textAlign': 'center'}),
            html.Div(children=[

                # Column 1
                html.Div([
                    html.B("Masse Molaire (g/mol)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-mass-mol', placeholder="Entrez la masse molaire", type='number',
                              value=Physics.ATM_SEA_LEVEL['m_mol'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Pression (Pa)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-pressure', placeholder="Entrez la pression", type='number',
                              value=Physics.ATM_SEA_LEVEL['pressure'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Température (K)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-temperature', placeholder="Entrez la température", type='number',
                              value=Physics.ATM_SEA_LEVEL['temperature'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),

                # Column 2
                html.Div([
                    html.B("Altitude (m)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-altitude', placeholder="Entrez l'altitude", type='number',
                              value=Physics.ATM_SEA_LEVEL['altitude'],
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Vitesse X", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-velocity-x', placeholder="Entrez la vitesse selon X", type='number',
                              value=0,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Vitesse Y", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-velocity-y', placeholder="Entrez la vitesse selon Y", type='number',
                              value=0,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Gravité (m.s2)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-gravity', placeholder="Entrez la gravité", type='number',
                              value=Physics.ATM_SEA_LEVEL['gravity'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),

                # Column 3
                html.Div([
                    html.B("Densité", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-density', placeholder="Entrez la densité", type='number',
                              value=Physics.ATM_SEA_LEVEL['density'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Viscosité (Pa.s)", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-viscosity', placeholder="Entrez la viscosité", type='number',
                              value=Physics.ATM_SEA_LEVEL['viscosity'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Indice Adiabatique", style={'color': '#f0f0f0'}),
                    dcc.Input(id='input-gamma', placeholder="Entrez l'indice adiabatique", type='number',
                              value=Physics.ATM_SEA_LEVEL['gamma'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'}),
            html.Div(children=[
                html.Button("OK", id='ok-button-value', className='ok-button', n_clicks=0),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '40%',
                      'margin': '0 auto', 'gap': '15px', 'margin-top': '100px'}),

        ]),
        # Third tab : Results
        dcc.Tab(label='Calculs', value='tab-3', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
                selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            html.H1(children='Résultats', style={'color': '#f0f0f0', 'textAlign': 'center'}),
            html.Div(children=[
                html.Button("Calcul", id='ok-button-calcul', className='ok-button', n_clicks=0),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '40%',
                      'margin': '0 auto', 'gap': '15px'}),
            dcc.Graph(id='results-graphs', figure=go.Figure(layout=dark_graph_layout), style={'height': '100vh'})
        ]),

    ]),
])