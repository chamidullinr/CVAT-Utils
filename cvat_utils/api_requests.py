import json
import logging
import os
from typing import Union

import requests
from dotenv import load_dotenv

logger = logging.getLogger("cvat_utils")

_username = None
_password = None


def _load_content(resp: requests.Response) -> Union[dict, bytes]:
    """Load JSON content from Response object.

    The method will return None if the status code is not 200, 201, or 202.

    Parameters
    ----------
    resp
        A Response object.

    Returns
    -------
    A Response content as dictionary or bytes.
    """
    # get JSON content
    try:
        content = json.loads(resp.content)
    except Exception:
        logger.error("Failed to load Response content as JSON.")
        content = resp.content

    if resp.status_code not in [200, 201, 202]:
        # print raw content
        out = None
        logger.warning(f"Status code reason: {resp.reason}")
        logger.warning(f"Content: {content}")
    else:
        out = content
    return out


def load_credentials():
    """Load CVAT credentials from .env file or environment variables.

    Username should be stored as `CVAT_USERNAME`.
    Password should be stored as `CVAT_PASSWORD`.
    """
    global _username, _password
    if _username is None or _password is None:
        # get environment variables from .env
        is_success = load_dotenv()
        if not is_success:
            is_success = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
        logger.debug(f"Loaded environment files from `.env` file: {is_success}")

        # get CVAT credentials from environment variables
        _username = os.getenv("CVAT_USERNAME")
        _password = os.getenv("CVAT_PASSWORD")

        if _username is None:
            err = "Environment variable 'CVAT_USERNAME' is not set."
            if not is_success:
                err += " Did not find any `.env` file."
            raise ValueError(err)

        if _password is None:
            err = "Environment variable 'CVAT_PASSWORD' is not set."
            if not is_success:
                err += " Did not find any `.env` file."
            raise ValueError(err)


def request(
    method: str, url: str, load_content: bool = True, **kwargs
) -> Union[dict, bytes, requests.Response]:
    """Create HTTP request.

    Parameters
    ----------
    method
        Method for the new Request object.
        One of `GET`, `OPTIONS`, `HEAD`, `POST`, `PUT`, `PATCH`, or `DELETE`.
    url
        URL for the new Request object.
    load_content
        If true return JSON content from Response object.
    kwargs
        Additional key arguments passed to `requests.get` method.

    Returns
    -------
    A Response content as a dictionary or bytes if load_content is true,
    otherwise a Response object.
    """
    # load credentials from .env file or environment variables
    load_credentials()

    # call request
    resp = requests.request(method, url, auth=(_username, _password), **kwargs)
    logger.debug(
        f"Received response with status code {resp.status_code} "
        f"for {method} request with url: {resp.url}"
    )
    return _load_content(resp) if load_content else resp


def get(url: str, load_content: bool = True, **kwargs) -> Union[dict, bytes, requests.Response]:
    """Create GET request to get data from CVAT."""
    return request("GET", url, load_content, **kwargs)


def patch(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create PATCH request to update records in CVAT."""
    return request("PATCH", url, load_content, params=params, json=data, **kwargs)


def put(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create PUT request to update records in CVAT."""
    return request("PUT", url, load_content, params=params, json=data, **kwargs)


def post(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create POST request to update records in CVAT."""
    return request("POST", url, load_content, params=params, json=data, **kwargs)
