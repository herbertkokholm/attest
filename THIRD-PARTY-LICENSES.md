# Third-party licenses

This file lists the licenses of attest's third-party Python dependencies,
including all optional vendor SDKs (`pip install -e ".[all]"`). It does not
include development-only tooling (`ruff`, `mypy`, `pytest`, ...), which is
never distributed with the software.

Generated with [`pip-licenses`](https://github.com/raimon49/pip-licenses):

```sh
pip install -e ".[all]" pip-licenses
pip-licenses --format=markdown --with-urls --order=name --ignore-packages attest --from=classifier
```

A license of `UNKNOWN` means the package does not declare a license
classifier; follow the URL to the package's repository to check its actual
license terms.

<!-- BEGIN GENERATED TABLE -->
| Name                               | Version         | License                                            | URL                                                                                                 |
|------------------------------------|-----------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Pygments                           | 2.20.0          | BSD-2-Clause                                       | https://pygments.org                                                                                |
| aiohappyeyeballs                   | 2.7.1           | Python Software Foundation License                 | https://github.com/aio-libs/aiohappyeyeballs                                                        |
| aiohttp                            | 3.14.3          | UNKNOWN                                            | https://github.com/aio-libs/aiohttp                                                                 |
| aiosignal                          | 1.4.0           | Apache Software License                            | https://github.com/aio-libs/aiosignal                                                               |
| annotated-types                    | 0.8.0           | MIT                                                | https://github.com/annotated-types/annotated-types                                                  |
| anthropic                          | 0.122.0         | MIT License                                        | https://github.com/anthropics/anthropic-sdk-python                                                  |
| anyio                              | 4.14.2          | MIT                                                | https://anyio.readthedocs.io/en/stable/versionhistory.html                                          |
| attrs                              | 26.1.0          | MIT                                                | https://www.attrs.org/en/stable/changelog.html                                                      |
| certifi                            | 2026.7.22       | Mozilla Public License 2.0 (MPL 2.0)               | https://github.com/certifi/python-certifi                                                           |
| cffi                               | 2.1.1           | MIT-0                                              | https://cffi.readthedocs.io/en/latest/whatsnew.html                                                 |
| charset-normalizer                 | 3.5.1           | UNKNOWN                                            | https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md                                |
| cryptography                       | 50.0.0          | Apache-2.0 OR BSD-3-Clause                         | https://github.com/pyca/cryptography                                                                |
| cyclopts                           | 4.22.5          | Apache-2.0                                         | https://github.com/BrianPugh/cyclopts                                                               |
| detect_agent                       | 0.6.0           | UNKNOWN                                            | https://github.com/togethercomputer/detect_agent                                                    |
| distro                             | 1.9.0           | Apache Software License                            | https://github.com/python-distro/distro                                                             |
| docstring_parser                   | 0.18.0          | MIT License                                        | https://github.com/rr-/docstring_parser                                                             |
| eval_type_backport                 | 0.4.0           | MIT License                                        | https://github.com/alexmojaki/eval_type_backport                                                    |
| filelock                           | 3.32.3          | MIT                                                | https://github.com/tox-dev/py-filelock                                                              |
| fireworks-ai                       | 1.2.9           | Apache Software License                            | https://github.com/fw-ai-external/python-sdk                                                        |
| frozenlist                         | 1.8.0           | UNKNOWN                                            | https://github.com/aio-libs/frozenlist                                                              |
| google-ai-generativelanguage       | 0.6.15          | Apache Software License                            | https://github.com/googleapis/google-cloud-python/tree/main/packages/google-ai-generativelanguage   |
| google-api-core                    | 2.33.0          | Apache Software License                            | https://github.com/googleapis/google-cloud-python/tree/main/packages/google-api-core                |
| google-api-python-client           | 2.198.0         | Apache Software License                            | https://github.com/googleapis/google-api-python-client/                                             |
| google-auth                        | 2.56.3          | Apache Software License                            | https://github.com/googleapis/google-cloud-python/tree/main/packages/google-auth                    |
| google-auth-httplib2               | 0.4.1           | Apache Software License                            | https://github.com/googleapis/google-cloud-python/packages/google-auth-httplib2                     |
| google-generativeai                | 0.8.6           | Apache Software License                            | https://github.com/google/generative-ai-python                                                      |
| googleapis-common-protos           | 1.75.0          | Apache Software License                            | https://github.com/googleapis/google-cloud-python/tree/main/packages/googleapis-common-protos       |
| grpcio                             | 1.83.0          | Apache-2.0                                         | https://grpc.io                                                                                     |
| grpcio-status                      | 1.71.2          | Apache Software License                            | https://grpc.io                                                                                     |
| h11                                | 0.16.0          | MIT License                                        | https://github.com/python-hyper/h11                                                                 |
| httpcore                           | 1.0.9           | BSD-3-Clause                                       | https://www.encode.io/httpcore/                                                                     |
| httpcore2                          | 2.10.0          | BSD-3-Clause                                       | https://github.com/pydantic/httpx2                                                                  |
| httplib2                           | 0.32.0          | MIT License                                        | https://github.com/httplib2/httplib2                                                                |
| httpx                              | 0.28.1          | BSD License                                        | https://github.com/encode/httpx                                                                     |
| httpx-aiohttp                      | 0.2.0           | UNKNOWN                                            | https://karpetrosyan.github.io/httpx-aiohttp/                                                       |
| httpx2                             | 2.10.0          | BSD-3-Clause                                       | https://github.com/pydantic/httpx2                                                                  |
| idna                               | 3.18            | BSD-3-Clause                                       | https://github.com/kjd/idna                                                                         |
| jiter                              | 0.16.0          | MIT                                                | https://github.com/pydantic/jiter/                                                                  |
| jsonpath-python                    | 1.1.6           | MIT License                                        | https://github.com/sean2077/jsonpath-python                                                         |
| krippendorff                       | 0.8.2           | GPL-3.0-or-later                                   | https://github.com/pln-fing-udelar/fast-krippendorff                                                |
| markdown-it-py                     | 4.2.0           | MIT License                                        | https://github.com/executablebooks/markdown-it-py                                                   |
| mdurl                              | 0.1.2           | MIT License                                        | https://github.com/executablebooks/mdurl                                                            |
| mistralai                          | 2.9.3           | UNKNOWN                                            | https://github.com/mistralai/client-python.git                                                      |
| multidict                          | 6.7.1           | UNKNOWN                                            | https://github.com/aio-libs/multidict                                                               |
| numpy                              | 2.4.6           | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org                                                                                   |
| openai                             | 3.1.0           | Apache Software License                            | https://github.com/openai/openai-python                                                             |
| opentelemetry-api                  | 1.44.0          | Apache-2.0                                         | https://github.com/open-telemetry/opentelemetry-python/tree/main/opentelemetry-api                  |
| opentelemetry-semantic-conventions | 0.65b0          | Apache-2.0                                         | https://github.com/open-telemetry/opentelemetry-python/tree/main/opentelemetry-semantic-conventions |
| propcache                          | 0.5.2           | Apache Software License                            | https://github.com/aio-libs/propcache                                                               |
| proto-plus                         | 1.28.2          | Apache Software License                            | https://github.com/googleapis/google-cloud-python/tree/main/packages/proto-plus                     |
| protobuf                           | 5.29.6          | UNKNOWN                                            | https://developers.google.com/protocol-buffers/                                                     |
| pyasn1                             | 0.6.4           | UNKNOWN                                            | https://github.com/pyasn1/pyasn1                                                                    |
| pyasn1_modules                     | 0.4.2           | BSD License                                        | https://github.com/pyasn1/pyasn1-modules                                                            |
| pycparser                          | 3.0             | BSD-3-Clause                                       | https://github.com/eliben/pycparser                                                                 |
| pydantic                           | 2.12.5          | MIT                                                | https://github.com/pydantic/pydantic                                                                |
| pydantic_core                      | 2.41.5          | MIT                                                | https://github.com/pydantic/pydantic-core                                                           |
| pyparsing                          | 3.3.2           | MIT                                                | https://github.com/pyparsing/pyparsing/                                                             |
| python-dateutil                    | 2.9.0.post0     | Apache Software License; BSD License               | https://github.com/dateutil/dateutil                                                                |
| requests                           | 2.34.2          | Apache Software License                            | https://github.com/psf/requests                                                                     |
| rich                               | 15.0.0          | MIT License                                        | https://github.com/Textualize/rich                                                                  |
| rich-rst                           | 2.1.0           | MIT                                                | https://wasi-master.github.io/rich-rst                                                              |
| scipy                              | 1.17.1          | BSD License                                        | https://scipy.org/                                                                                  |
| six                                | 1.17.0          | MIT License                                        | https://github.com/benjaminp/six                                                                    |
| sniffio                            | 1.3.1           | Apache Software License; MIT License               | https://github.com/python-trio/sniffio                                                              |
| together                           | 2.31.0          | Apache Software License                            | https://github.com/togethercomputer/together-py                                                     |
| tqdm                               | 4.70.0          | UNKNOWN                                            | https://tqdm.github.io                                                                              |
| truststore                         | 0.10.4          | MIT                                                | https://github.com/sethmlarson/truststore                                                           |
| types-PyYAML                       | 6.0.12.20260815 | Apache-2.0                                         | https://github.com/python/typeshed                                                                  |
| types-tqdm                         | 4.70.0.20260805 | Apache-2.0                                         | https://github.com/python/typeshed                                                                  |
| typing-inspection                  | 0.4.4           | MIT                                                | https://github.com/pydantic/typing-inspection                                                       |
| typing_extensions                  | 4.16.0          | PSF-2.0                                            | https://github.com/python/typing_extensions                                                         |
| uritemplate                        | 4.2.0           | UNKNOWN                                            | https://uritemplate.readthedocs.org                                                                 |
| urllib3                            | 2.7.0           | MIT                                                | https://github.com/urllib3/urllib3/blob/main/CHANGES.rst                                            |
| yarl                               | 1.24.5          | UNKNOWN                                            | https://github.com/aio-libs/yarl                                                                    |
<!-- END GENERATED TABLE -->

## Manually verified `UNKNOWN` entries

`pip-licenses` only reads a package's `License` metadata field and trove
classifiers; it does not look inside the bundled `LICENSE` file in the
wheel's `dist-info/licenses/` directory. The following packages report
`UNKNOWN` above for that reason but have a confirmed license, checked
directly against the installed wheel on 2026-08-15:

| Name           | Confirmed license | Evidence                                                                          |
|----------------|--------------------|------------------------------------------------------------------------------------|
| `mistralai`    | Apache-2.0         | `mistralai-2.9.3.dist-info/licenses/LICENSE`, full Apache 2.0 text                 |
| `detect_agent` | Apache-2.0         | `detect_agent-0.5.0.dist-info/licenses/LICENSE`, full Apache 2.0 text              |
| `httpx-aiohttp`| BSD-3-Clause       | `License` metadata field contains the full 3-clause BSD text verbatim              |

This note is a manual addition, not auto-generated -- it survives
`tools/gen_third_party_licenses.py` reruns because it sits after the
`END GENERATED TABLE` marker. Re-verify if these three packages' versions
change materially.

The remaining `UNKNOWN` rows above (`aiohttp`, `charset-normalizer`,
`frozenlist`, `multidict`, `protobuf`, `pyasn1`, `tqdm`, `uritemplate`,
`yarl`) have not had the same bundled-`LICENSE`-file check applied; they are
widely known to carry permissive or Apache-2.0 licenses, but that has not
been individually confirmed here.
