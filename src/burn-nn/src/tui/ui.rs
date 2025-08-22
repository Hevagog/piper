use crate::tui::app::{CurrentScreen, Visualizer};
use ratatui::{
    prelude::*,
    symbols,
    widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph},
};

pub fn ui<B: Backend>(f: &mut Frame, app: &Visualizer) {
    let chunks = Layout::default()
        .constraints([Constraint::Percentage(100)])
        .split(f.area());

    match app.current_screen {
        CurrentScreen::Main => main_screen::<B>(f, chunks[0]),
        CurrentScreen::Exiting => exiting_screen::<B>(f, chunks[0]),
        CurrentScreen::TrainingView => training_screen::<B>(f, app, chunks[0]),
    }
}

fn main_screen<B: Backend>(f: &mut Frame, area: Rect) {
    let text = vec![
        Line::from("Welcome to the Training Visualizer".bold()),
        Line::from(""),
        Line::from("Press 't' to view the training process."),
        Line::from("Press 'q' to quit."),
    ];
    let paragraph = Paragraph::new(text)
        .block(Block::bordered().title("Main Menu"))
        .alignment(Alignment::Center);
    f.render_widget(paragraph, area);
}

fn exiting_screen<B: Backend>(f: &mut Frame, area: Rect) {
    let text = "Are you sure you want to quit? (y/n)";
    let paragraph = Paragraph::new(text)
        .block(Block::bordered().title("Exit"))
        .alignment(Alignment::Center);
    f.render_widget(paragraph, area);
}

fn training_screen<B: Backend>(f: &mut Frame, app: &Visualizer, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(area);

    render_loss_chart::<B>(f, app, chunks[0]);
    render_accuracy_chart::<B>(f, app, chunks[1]);
}

fn render_loss_chart<B: Backend>(f: &mut Frame, app: &Visualizer, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Red))
            .graph_type(GraphType::Line)
            .data(&app.loss),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title("Training Loss"))
        .x_axis(
            Axis::default()
                .title("Epoch")
                .style(Style::default().fg(Color::Gray))
                .bounds(app.window),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 100.0]),
        );

    f.render_widget(chart, area);
}

fn render_accuracy_chart<B: Backend>(f: &mut Frame, app: &Visualizer, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Accuracy")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Green))
            .graph_type(GraphType::Line)
            .data(&app.accuracy),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title("Training Accuracy"))
        .x_axis(
            Axis::default()
                .title("Epoch")
                .style(Style::default().fg(Color::Gray))
                .bounds(app.window),
        )
        .y_axis(
            Axis::default()
                .title("Accuracy")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 1.0]),
        );

    f.render_widget(chart, area);
}
