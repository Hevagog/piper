use color_eyre::{Result, eyre::Ok};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{Terminal, prelude::*};
use std::time::{Duration, Instant};

use crate::tui::ui::ui;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CurrentScreen {
    #[default]
    Main,
    Exiting,
    TrainingView,
}

#[derive(Debug)]
pub struct Visualizer {
    pub exit: bool,
    pub current_screen: CurrentScreen,
    pub loss: Vec<(f64, f64)>,
    pub accuracy: Vec<(f64, f64)>,
    pub window: [f64; 2],
    tick_count: f64,
}

impl Visualizer {
    pub fn new() -> Self {
        let loss = (0..100)
            .map(|i| (i as f64, 100.0 * (-i as f64 / 100.0).exp()))
            .collect();
        let accuracy = (0..100)
            .map(|i| (i as f64, 1.0 - (-i as f64 / 100.0).exp()))
            .collect();
        Self {
            exit: false,
            current_screen: CurrentScreen::default(),
            loss,
            accuracy,
            window: [0.0, 100.0],
            tick_count: 0.0,
        }
    }

    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        let tick_rate = Duration::from_millis(250);
        let mut last_tick = Instant::now();
        while !self.exit {
            terminal.draw(|frame| self.render::<B>(frame))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key_event(key)?;
                }
            }

            if last_tick.elapsed() >= tick_rate {
                self.on_tick();
                last_tick = Instant::now();
            }
        }
        Ok(())
    }

    fn on_tick(&mut self) {
        self.tick_count += 1.0;
        self.window[0] += 1.0;
        self.window[1] += 1.0;

        let new_loss_point = (
            self.tick_count + 100.0,
            100.0 * (-(self.tick_count) / 100.0).exp(),
        );
        self.loss.push(new_loss_point);

        let new_accuracy_point = (
            self.tick_count + 100.0,
            1.0 - (-(self.tick_count) / 100.0).exp(),
        );
        self.accuracy.push(new_accuracy_point);
    }

    fn render<B: Backend>(&self, frame: &mut Frame) {
        ui::<B>(frame, self);
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> Result<()> {
        if key_event.kind != KeyEventKind::Press {
            return Ok(());
        }
        match self.current_screen {
            CurrentScreen::Exiting => match key_event.code {
                KeyCode::Char('y') => {
                    self.exit = true;
                }
                KeyCode::Char('n') | KeyCode::Esc => {
                    self.current_screen = CurrentScreen::Main;
                }
                _ => {}
            },
            CurrentScreen::Main => match key_event.code {
                KeyCode::Char('q') => {
                    self.current_screen = CurrentScreen::Exiting;
                }
                KeyCode::Char('t') => {
                    self.current_screen = CurrentScreen::TrainingView;
                }
                _ => {}
            },
            CurrentScreen::TrainingView => match key_event.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    self.current_screen = CurrentScreen::Main;
                }
                _ => {}
            },
        }
        Ok(())
    }
}
