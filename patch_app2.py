import re

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

for path in ['src/app.rs', 'src/worker.rs']:
    with open(path, 'r') as f:
        content = f.read()
    
    # remove old macro if it exists
    content = re.sub(r'macro_rules! file_log \{.*?\}\s*', '', content, flags=re.DOTALL)
    
    # put it at the very top, after the first line or just top
    content = log_macro + "\n" + content
    
    with open(path, 'w') as f:
        f.write(content)
