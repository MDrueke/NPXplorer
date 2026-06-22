import re

with open('src/app.rs', 'r') as f:
    content = f.read()

# Replace println! with file logging
log_macro = """
macro_rules! file_log {
    ($($arg:tt)*) => {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("debug.log") {
            use std::io::Write;
            let _ = writeln!(f, $($arg)*);
        }
    };
}
"""

content = content.replace("use egui::Color32;", "use egui::Color32;\n" + log_macro)
content = content.replace("println!(", "file_log!(")

with open('src/app.rs', 'w') as f:
    f.write(content)

with open('src/worker.rs', 'r') as f:
    content = f.read()

content = content.replace("use std::sync::Arc;", "use std::sync::Arc;\n" + log_macro)
content = content.replace("println!(", "file_log!(")

with open('src/worker.rs', 'w') as f:
    f.write(content)
