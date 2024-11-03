from dash import html, dcc
import plotly.graph_objects as go
from src.objects import Physics


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
    dcc.Store(id='calcul-store'),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Création du profil', value='tab-1', className='tab', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
                 selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            html.H1(children='Création du profil'),
                html.Div(children=[
                    html.Div(children=[
                        dcc.Dropdown(['Parabolique', 'Conique', 'Ariane 4'], id='dropdown-shape'),
                        html.Div(children=[
                            html.Button("OK", id='ok-button-profile', className='button', n_clicks=0),
                            html.Button("?", id="help-button-profile", className="help-button", n_clicks=0),
                        ], id='div-ok-help-button'),
                        dcc.Upload(id='upload-profile', children=html.Div([
                                'Glisser-Déposer ou ',
                                html.A('Selectionner un fichier')
                                ])
                        ),
                        html.P("", id='error-text-upload', style={'visibility': 'hidden'}),
                    ], id='div-dropdown'),
                    html.Div(children=[
                        html.B("Taille du profil", id='text-input-length'),
                        dcc.Input(id='input-length', placeholder='Entrez la taille du profil', type='number'),
                        html.B("Angle du profil", id='text-input-angle'),
                        dcc.Input(id='input-angle', placeholder="Entrez l'angle du profil", type='number'),
                        html.Div(children=[], id='div-dynamic-components'),
                    ], id='div-selection-attributes'),
                ], id='div-selection'),
                dcc.Graph(id='shape-graphs', figure=go.Figure(layout=dark_graph_layout))
        ]),

        # Second Tab: Valeurs initiales
        dcc.Tab(label='Valeurs initiales', value='tab-2', className='tab', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
            selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            dcc.Checklist(options=[{'label': 'Rendre les valeurs modifiables', 'value': 'modify'}], id='modify-values-check'),
            html.H1(children='Valeurs initiales'),
            html.Div(children=[
                # Column 1
                html.Div([
                    html.B("Masse Molaire (g/mol)"),
                    dcc.Input(id='input-mass-mol', placeholder="Entrez la masse molaire", type='number',
                              value=Physics.ATM_SEA_LEVEL['m_mol'], disabled=True,
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Pression (Pa)"),
                    dcc.Input(id='input-pressure', placeholder="Entrez la pression", type='number',
                              value=Physics.ATM_SEA_LEVEL['pressure'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Température (K)"),
                    dcc.Input(id='input-temperature', placeholder="Entrez la température", type='number',
                              value=Physics.ATM_SEA_LEVEL['temperature'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),

                # Column 2
                html.Div([
                    html.B("Altitude (m)"),
                    dcc.Input(id='input-altitude', placeholder="Entrez l'altitude", type='number',
                              value=Physics.ATM_SEA_LEVEL['altitude'], 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Vitesse X"),
                    dcc.Input(id='input-velocity-x', placeholder="Entrez la vitesse selon X", type='number',
                              value=0, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Vitesse Y"),
                    dcc.Input(id='input-velocity-y', placeholder="Entrez la vitesse selon Y", type='number',
                              value=0, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                    html.B("Gravité (m.s2)"),
                    dcc.Input(id='input-gravity', placeholder="Entrez la gravité", type='number',
                              value=Physics.ATM_SEA_LEVEL['gravity'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),

                # Column 3
                html.Div([
                    html.B("Densité"),
                    dcc.Input(id='input-density', placeholder="Entrez la densité", type='number',
                              value=Physics.ATM_SEA_LEVEL['density'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Viscosité (Pa.s)"),
                    dcc.Input(id='input-viscosity', placeholder="Entrez la viscosité", type='number',
                              value=Physics.ATM_SEA_LEVEL['viscosity'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),

                    html.B("Indice Adiabatique"),
                    dcc.Input(id='input-gamma', placeholder="Entrez l'indice adiabatique", type='number',
                              value=Physics.ATM_SEA_LEVEL['gamma'], disabled=True, 
                              style={'backgroundColor': '#3a3a3a', 'color': '#f0f0f0'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}),
            ], id='div-initial-values'),
            html.Div(children=[
                html.Button("OK", id='ok-button-value', className='button', n_clicks=0),
            ], className='div-button-centered'),
        ]),
        # Third tab : Results
        dcc.Tab(label='Calculs', value='tab-3', className='tab', style={'backgroundColor': '#2c2f33', 'color': '#f0f0f0'},
                selected_style={'backgroundColor': '#3a3a3a', 'color': '#ffffff'}, children=[
            html.H1(children='Résultats'),
            html.Div(children=[
                html.Button("Calcul", className='button', id='ok-button-calcul'),
            ], className='div-button-centered'),
            dcc.Store(id='highlight-store', data=None),
            html.Div(children=[], style={'visibility': 'hidden', 'border-radius': '10px'}, id='highlighted-graph'),
            html.Div(children=[
                html.Div(
                    id={'type': 'grid-item', 'index': i},
                    children=[
                        dcc.Loading(id={'type': 'loading', 'index': i}, type='circle', color='#4caf50', children=[
                            html.Img(style={'backgroundColor': '#2c2f33'}, id={'type': 'image', 'index': i}),
                            dcc.Graph(className='graph-grid', id={'type': 'graph-grid', 'index': i},
                                      config={'staticPlot': True}, figure=go.Figure(layout=dark_graph_layout))
                        ])
                    ],
                    className='grid-item') for i in range(12)
            ], id='grid-container'),
        ]),
    ]),
])