mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = falsn\n
headless = true\n\n
\n\
" > ~/.streamlit/config.toml
