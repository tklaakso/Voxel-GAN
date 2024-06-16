import requests as req
from bs4 import BeautifulSoup
import json
import base64
import os
import traceback
import sqlite3

def get_extension(img_path):
    if '.' in img_path and len(img_path[img_path.rindex('.') + 1:]) <= 6:
        return img_path[img_path.rindex('.') + 1:]
    return 'png'

def flush_cache(data):
    con = sqlite3.connect('voxel.db')
    with con:
        cur = con.cursor()
        cur.executemany('insert or ignore into build values(?, ?, ?, ?, ?)', [(k, v['title'], v['image_link'], v['image_data'], v['script_link']) for k, v in data.items()])
        blocks = []
        materials = {}
        for k, v in data.items():
            for block in v['blocks']:
                blocks.append({'build_id' : k, 'mat_id' : block['mat_id'], 'x' : block['x'], 'y' : block['y'], 'z' : block['z']})
                materials[block['mat_id']] = {'hex' : block['hex'], 'r' : block['rgb'][0], 'g' : block['rgb'][1], 'b' : block['rgb'][2], 'name' : block['name'], 'file' : block['file'], 'transparent' : block['transparent'], 'opacity' : block['opacity'], 'texture' : block['texture']}
        cur.executemany('insert or ignore into material values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [(k, v['hex'], v['r'], v['g'], v['b'], v['name'], v['file'], v['transparent'], v['opacity'], v['texture']) for k, v in materials.items()])
        cur.executemany('insert or ignore into block values(?, ?, ?, ?, ?)', [(x['build_id'], x['mat_id'], x['x'], x['y'], x['z']) for x in blocks])
    data.clear()

if __name__ == '__main__':
    data = {}
    image_out = 'build_images'
    if not os.path.exists(image_out):
        os.makedirs(image_out)
    if os.path.exists('data.json'):
        with open('data.json', 'r') as file:
            data = json.load(file)
    main_page = req.get('https://www.grabcraft.com/minecraft/houses').content
    soup = BeautifulSoup(main_page, 'html.parser')
    li = soup.find('li', {'class' : 'cats-10 level_2 selected'})
    categories = li.ul.find_all('li', recursive = False)
    links = ['https://www.grabcraft.com' + c.a['href'] for c in categories]
    num_fetched = 0
    for link in links:
        page_num = 1
        while True:
            subpage = req.get(link + '/pg/' + str(page_num)).content
            page_num += 1
            subsoup = BeautifulSoup(subpage, 'html.parser')
            all_builds = subsoup.find('div', {'class' : 'products row-size-4'})
            if not all_builds:
                break
            builds = all_builds.find_all('div', recursive = False)
            if not builds:
                break
            build_info = [('https://www.grabcraft.com' + b.find('h3', {'class' : 'name'}).a['href'], b.find('h3', {'class' : 'name'}).a['title'], b.find('img')['src']) for b in builds]
            for build_link, title, image_link in build_info:
                try:
                    build_id = build_link[build_link.index('minecraft/') + len('minecraft/'):]
                    build_id = build_id[:build_id.index('/')]
                    if build_id in data:
                        continue
                    print('Fetching: ' + title)
                    num_fetched += 1
                    build_image = req.get(image_link).content
                    with open(image_out + '/' + build_id + '.' + get_extension(image_link), 'wb') as file:
                        file.write(build_image)
                    build_page = req.get(build_link).content
                    build_soup = BeautifulSoup(build_page, 'html.parser')
                    script_link = build_soup.find('script', {'src' : lambda x: x and x.startswith('https://www.grabcraft.com/js/RenderObject')})['src']
                    script_page = req.get(script_link).content.decode('utf-8')
                    script_obj = json.loads(script_page[script_page.index('{'):])
                    blocks = []
                    for v1 in script_obj.values():
                        for v2 in v1.values():
                            for v3 in v2.values():
                                blocks.append(v3)
                    data[build_id] = {'title' : title, 'image_link' : image_link, 'image_data' : base64.b64encode(build_image).decode('utf-8'), 'script_link' : script_link, 'blocks' : blocks}
                    if num_fetched % 10 == 0:
                        flush_cache(data)
                except Exception:
                    print(traceback.format_exc())
    flush_cache(data)