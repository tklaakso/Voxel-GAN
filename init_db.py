import sqlite3

con = sqlite3.connect('voxel.db')

cur = con.cursor()
cur.execute('create table material(id int primary key, hex text, r int, g int, b int, name text, file text, transparent int, opacity int, texture text)')
cur.execute('create table build(id text primary key, title text, image_link text, image_data text, script_link text)')
cur.execute('create table block(build_id text, mat_id int, x int, y int, z int, foreign key(mat_id) references material(id), foreign key(build_id) references build(id), primary key(build_id, x, y, z))')

con.commit()