import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ...core.logger import jm_log

class RemoteAPIBase:
    """通用远程 API 请求基类"""
    
    def __init__(self):
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def post_request(self, url, headers, data, stream=False, timeout=60, verify=True):
        """统一的 POST 请求处理"""
        try:
            return self.session.post(url, headers=headers, json=data, stream=stream, timeout=timeout, verify=verify)
        except Exception as e:
            jm_log("ERROR", "RemoteAPI", f"POST 请求异常: {str(e)}")
            raise e

    def get_request(self, url, headers=None, stream=False, timeout=60, verify=True):
        """统一的 GET 请求处理"""
        try:
            return self.session.get(url, headers=headers, stream=stream, timeout=timeout, verify=verify)
        except Exception as e:
            jm_log("ERROR", "RemoteAPI", f"GET 请求异常: {str(e)}")
            raise e
