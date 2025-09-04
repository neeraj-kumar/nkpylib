"""Various helpful constants"""

import re

# as of apr 29, 2025
USER_AGENT = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0'

OLD_URL_REGEXP = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)

# very simple url regexp
URL_REGEXP = re.compile(r"https?:\/\/\S+")
