BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

https_proxy= http_proxy= all_proxy= streamlit run demo/home.py
