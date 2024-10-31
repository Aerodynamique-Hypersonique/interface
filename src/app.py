from src.tab1_profile.callbacks import *
from src.tab2_values.callbacks import *
from src.tab3_calculs.callbacks import *
from dash import Dash

app = Dash(suppress_callback_exceptions=True)
app.layout = get_layout() # call the function from layout file

define_callbacks1(app) # Callbacks tab 1
define_callbacks2(app) # Callbacks tab 2
define_callbacks3(app) # Callbacks tab 3

if __name__ == '__main__':
    app.run(debug=True)

