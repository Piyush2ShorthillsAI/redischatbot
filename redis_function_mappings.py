import asyncio
import sys
sys.path.append('src')
from src.tools.redis_query_engine import create_vector_index_hash, vector_search_hash, get_index_info
from src.tools.hash import hset, hgetall, set_vector_in_hash
from src.tools.misc import scan_all_keys

import asyncio

# Functions from the 'hash.py' module
from src.tools.hash import hset, hgetall, hget, hdel, hexists, set_vector_in_hash, get_vector_from_hash

# Functions from the 'json.py' module
from src.tools.json import json_set, json_get, json_del

# Functions from the 'list.py' module
from src.tools.list import lpush, rpush, lpop, rpop, lrange, llen

# Functions from the 'misc.py' module
from src.tools.misc import scan_all_keys, scan_keys, delete, type, expire, rename
from src.tools.server_management import dbsize, info, client_list
# Functions from the 'pub_sub.py' module
from src.tools.pub_sub import publish, subscribe, unsubscribe

# Functions from the 'redis_query_engine.py' module
from src.tools.redis_query_engine import create_vector_index_hash, vector_search_hash, get_index_info, get_indexes, get_indexed_keys_number

# Functions from the 'set.py' module
from src.tools.set import sadd, srem, smembers

# Functions from the 'sorted_set.py' module
from src.tools.sorted_set import zadd, zrange, zrem

# Functions from the 'stream.py' module
from src.tools.stream import xadd, xrange, xdel



# function_mappings = {
#     "delete": lambda key: asyncio.run(delete(key)),
#     "type": lambda key: asyncio.run(type(key)),
#     "expire": lambda key, seconds: asyncio.run(expire(key, seconds)),
#     "rename": lambda old_key, new_key: asyncio.run(rename(old_key, new_key)),
#     "scan_keys": lambda pattern='*': asyncio.run(scan_keys(pattern)),
#     "scan_all_keys": lambda prefix: asyncio.run(scan_all_keys(prefix)),
#     "zadd": lambda key, score, member: asyncio.run(zadd(key, score, member)),
#     "zrange": lambda key, start, end: asyncio.run(zrange(key, start, end)),
#     "zrem": lambda key, member: asyncio.run(zrem(key, member)),
#     "lpush": lambda key, value: asyncio.run(lpush(key, value)),
#     "rpush": lambda key, value: asyncio.run(rpush(key, value)),
#     "lpop": lambda key: asyncio.run(lpop(key)),
#     "rpop": lambda key: asyncio.run(rpop(key)),
#     "lrange": lambda key, start, stop: asyncio.run(lrange(key, start, stop)),
#     "llen": lambda key: asyncio.run(llen(key)),
#     "get_indexes": lambda: asyncio.run(get_indexes()),
#     "get_index_info": lambda index_name: asyncio.run(get_index_info(index_name)),
#     "get_indexed_keys_number": lambda index_name: asyncio.run(get_indexed_keys_number(index_name)),
#     "create_vector_index_hash": lambda index_name, dim=1536: asyncio.run(create_vector_index_hash(index_name, dim)),
#     "vector_search_hash": lambda query_vector, k=5, index_name='vector_index': asyncio.run(vector_search_hash(query_vector, k, index_name)),
#     "hset": lambda key, field, value: asyncio.run(hset(key, field, value)),
#     "hget": lambda key, field: asyncio.run(hget(key, field)),
#     "hdel": lambda key, field: asyncio.run(hdel(key, field)),
#     "hgetall": lambda key: asyncio.run(hgetall(key)),
#     "hexists": lambda key, field: asyncio.run(hexists(key, field)),
#     "set_vector_in_hash": lambda key, vector: asyncio.run(set_vector_in_hash(key, vector)),
#     "get_vector_from_hash": lambda key: asyncio.run(get_vector_from_hash(key)),
#     "json_set": lambda key, path, value: asyncio.run(json_set(key, path, value)),
#     "json_get": lambda key, path='$': asyncio.run(json_get(key, path)),
#     "json_del": lambda key, path='$': asyncio.run(json_del(key, path)),
#     "dbsize": lambda: asyncio.run(dbsize()),
#     "info": lambda section='default': asyncio.run(info(section)),
#     "client_list": lambda: asyncio.run(client_list()),
#     "xadd": lambda key, fields: asyncio.run(xadd(key, fields)),
#     "xrange": lambda key, count=1: asyncio.run(xrange(key, count)),
#     "xdel": lambda key, entry_id: asyncio.run(xdel(key, entry_id)),
#     "publish": lambda channel, message: asyncio.run(publish(channel, message)),
#     "subscribe": lambda channel: asyncio.run(subscribe(channel)),
#     "unsubscribe": lambda channel: asyncio.run(unsubscribe(channel)),
#     "sadd": lambda key, value: asyncio.run(sadd(key, value)),
#     "srem": lambda key, value: asyncio.run(srem(key, value)),
#     "smembers": lambda key: asyncio.run(smembers(key)),
#     "set": lambda key, value: asyncio.run(set(key, value)),
#     "get": lambda key: asyncio.run(get(key))
# }


# filepath: /home/shtlp_0170/Downloads/mcp-redis-0.2.0 (copy)(2)/mcp-redis-0.2.0 (copy)/redis_function_mappings.py
function_mappings = {
    "delete": delete,
    "type": type,
    "expire": expire,
    "rename": rename,
    "scan_keys": scan_keys,
    "scan_all_keys": scan_all_keys,
    "zadd": zadd,
    "zrange": zrange,
    "zrem": zrem,
    "lpush": lpush,
    "rpush": rpush,
    "lpop": lpop,
    "rpop": rpop,
    "lrange": lrange,
    "llen": llen,
    "get_indexes": get_indexes,
    "get_index_info": get_index_info,
    "get_indexed_keys_number": get_indexed_keys_number,
    "create_vector_index_hash": create_vector_index_hash,
    "vector_search_hash": vector_search_hash,
    "hset": hset,
    "hget": hget,
    "hdel": hdel,
    "hgetall": hgetall,
    "hexists": hexists,
    "set_vector_in_hash": set_vector_in_hash,
    "get_vector_from_hash": get_vector_from_hash,
    "json_set": json_set,
    "json_get": json_get,
    "json_del": json_del,
    "dbsize": dbsize,
    "info": info,
    "client_list": client_list,
    "xadd": xadd,
    "xrange": xrange,
    "xdel": xdel,
    "publish": publish,
    "subscribe": subscribe,
    "unsubscribe": unsubscribe,
    "sadd": sadd,
    "srem": srem,
    "smembers": smembers,
    "set": set,
   
}