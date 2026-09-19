from shiny import App, ui, render, reactive

app_ui = ui.page_fluid(
    ui.input_action_button("btn", "Generate"),
    ui.output_ui("my_ui")
)

def server(input, output, session):
    @render.ui
    @reactive.event(input.btn)
    def my_ui():
        return ui.input_checkbox("test", "Test", value=True)

app = App(app_ui, server)
