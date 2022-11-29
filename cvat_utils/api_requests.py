import json
import logging
import os
from typing import Union

import requests
from dotenv import load_dotenv

logger = logging.getLogger("scripts")

load_dotenv()  # get environment variables from .env

USERNAME = os.getenv("CVAT_USERNAME")
PASSWORD = os.getenv("CVAT_PASSWORD")
assert USERNAME is not None, "Environment variable 'CVAT_USERNAME' is not set."
assert PASSWORD is not None, "Environment variable 'CVAT_PASSWORD' is not set."


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


def get(url: str, load_content: bool = True, **kwargs) -> Union[dict, bytes, requests.Response]:
    """Create GET request.

    Parameters
    ----------
    url
        URL for the new Request object.
    load_content
        If true return JSON content from Response object.
    kwargs
        Additional key arguments passed to `requests.get` method.

    Returns
    -------
    A Response content as dictionary or bytes if load_content is true,
    otherwise a Response object.
    """
    resp = requests.get(url, auth=(USERNAME, PASSWORD), **kwargs)
    logger.debug(
        f"Received response with status code {resp.status_code} "
        f"for GET request with url: {resp.url}"
    )
    return _load_content(resp) if load_content else resp


def patch(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create PATCH request to update records in CVAT.

    Parameters
    ----------
    url
        URL for the new Request object.
    params
        A dictionary to send in the query string for the Request.
    data
        JSON data to send in the body of the Request.
    load_content
        If true return JSON content from Response object.
    kwargs
        Additional key arguments passed to `requests.patch` method.

    Returns
    -------
    A Response content as dictionary or bytes if load_content is true,
    otherwise a Response object.
    """
    resp = requests.patch(
        url,
        params=params,
        json=data,
        auth=(USERNAME, PASSWORD),
        **kwargs,
    )
    logger.debug(
        f"Received response with status code {resp.status_code} "
        f"for PATCH request with url: {resp.url}"
    )
    return _load_content(resp) if load_content else resp


def put(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create PUT request to update records in CVAT.

    Parameters
    ----------
    url
        URL for the new Request object.
    params
        A dictionary to send in the query string for the Request.
    data
        JSON data to send in the body of the Request.
    load_content
        If true return JSON content from Response object.
    kwargs
        Additional key arguments passed to `requests.put` method.

    Returns
    -------
    A Response content as dictionary or bytes if load_content is true,
    otherwise a Response object.
    """
    resp = requests.put(
        url,
        params=params,
        json=data,
        auth=(USERNAME, PASSWORD),
        **kwargs,
    )
    logger.debug(
        f"Received response with status code {resp.status_code} "
        f"for PUT request with url: {resp.url}"
    )
    return _load_content(resp) if load_content else resp


def post(
    url: str,
    params: dict = None,
    data: dict = None,
    load_content: bool = True,
    **kwargs,
) -> Union[dict, bytes, requests.Response]:
    """Create POST request to update records in CVAT.

    Parameters
    ----------
    url
        URL for the new Request object.
    params
        A dictionary to send in the query string for the Request.
    data
        JSON data to send in the body of the Request.
    load_content
        If true return JSON content from Response object.
    kwargs
        Additional key arguments passed to `requests.post` method.

    Returns
    -------
    A Response content as dictionary or bytes if load_content is true,
    otherwise a Response object.
    """
    resp = requests.post(
        url,
        params=params,
        json=data,
        auth=(USERNAME, PASSWORD),
        **kwargs,
    )
    logger.debug(
        f"Received response with status code {resp.status_code} "
        f"for POST request with url: {resp.url}"
    )
    return _load_content(resp) if load_content else resp
