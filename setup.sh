mkdir -p ~/.streamlit/
echo "[theme]
base="dark"\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
