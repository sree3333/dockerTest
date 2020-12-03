import imp
import requests, json

def AWS_load(url):
    r = requests.get(url)
    f = (r.content)
    mymodule = imp.new_module('mymodule')
    exec(f, mymodule.__dict__)
    return mymodule

def check_merchant(merchantID, gender):
    url = "https://commonms.s3.ap-south-1.amazonaws.com/quickSize_utils/merchantlists/merchanlist.json"
    merchantlists = requests.get(url).json()
    if merchantID in merchantlists:
        preference = merchantlists[merchantID][gender]
    else:
        preference = {}
    return preference

def get_JSON(url):
    brand_info = requests.get(url).json()
    return brand_info