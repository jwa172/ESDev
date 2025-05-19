from dash import Dash, html, dash_table, Output, Input
import pandas as pd

# Sample data
df = pd.DataFrame({
    'Country': ['Canada', 'USA', 'Mexico'],
    'Population': [37, 328, 126],
})

# Initialize app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Editable Table Example"),
    
    dash_table.DataTable(
        id='editable-table',
        columns=[{'name': i, 'id': i, 'editable': True} for i in df.columns],  # ✅ mark columns as editable
        data=df.to_dict('records'),
        editable=True,       # ✅ global editable flag
        page_size=5
    ),
    
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('editable-table', 'data')
)
def display_changes(data):
    updated_df = pd.DataFrame(data)
    return f"Updated data: {updated_df.to_dict()}"

# Run in notebook or script
app.run(debug=True)
