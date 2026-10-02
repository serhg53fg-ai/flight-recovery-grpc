"""Render an unprivileged Nginx configuration into a controlled runtime tree."""
import argparse
from pathlib import Path


def safe_path(value):
    path = Path(value).resolve()
    text = str(path)
    if any(ord(char) < 32 or char in '"$;{}#\\' for char in text):
        raise ValueError('路径含不支持的配置字符')
    for original in map(Path, ('/srv/flight-example/user/tzb/tiaozhan4', '/srv/flight-example/user/rpc', '/srv/flight-example/user/grpc')):
        if path == original or original in path.parents:
            raise ValueError('部署路径不能位于原项目目录')
    return path


def validate_outputs(runtime):
    for name in ('body', 'proxy', 'fastcgi', 'uwsgi', 'scgi', 'access.log', 'nginx.pid', 'nginx.conf'):
        output = runtime / name
        if output.is_symlink():
            raise ValueError('运行输出不能为符号链接')
        safe_path(output)
        if output.is_dir() and any(child.is_symlink() for child in output.rglob('*')):
            raise ValueError('运行临时目录不能包含符号链接')
        if output.is_file() and output.stat().st_nlink > 1:
            raise ValueError('运行输出不能为硬链接')


def render_config(project_root, runtime_root, listen_port=8080, web_a_port=7101, web_b_port=7102):
    ports = (listen_port, web_a_port, web_b_port)
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports) or len(set(ports)) != 3:
        raise ValueError('部署须使用三个不同的非特权端口')
    project, runtime = safe_path(project_root), safe_path(runtime_root)
    validate_outputs(runtime)
    static = project / 'apps/flight/static'
    if not static.is_dir():
        raise ValueError('项目静态目录不存在')
    return f'''worker_processes 1;
daemon off;
pid "{runtime}/nginx.pid";
error_log stderr warn;
events {{ worker_connections 256; }}
http {{
    default_type application/octet-stream;
    types {{ application/javascript js; text/css css; application/json json; image/png png; image/svg+xml svg; }}
    log_format flight '$request_method $uri status=$status upstream=$upstream_addr upstream_status=$upstream_status';
    access_log "{runtime}/access.log" flight;
    client_body_temp_path "{runtime}/body";
    proxy_temp_path "{runtime}/proxy";
    fastcgi_temp_path "{runtime}/fastcgi";
    uwsgi_temp_path "{runtime}/uwsgi";
    scgi_temp_path "{runtime}/scgi";
    limit_conn_zone $binary_remote_addr zone=flight_sse:1m;
    upstream flight_web {{
        server 127.0.0.1:{web_a_port} max_fails=1 fail_timeout=2s;
        server 127.0.0.1:{web_b_port} max_fails=1 fail_timeout=2s;
        keepalive 16;
    }}
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 2s;
    proxy_send_timeout 90s;
    proxy_read_timeout 90s;
    proxy_next_upstream error timeout http_502 http_503 http_504;
    proxy_next_upstream_tries 2;
    proxy_next_upstream_timeout 5s;
    server {{
        listen 127.0.0.1:{listen_port};
        server_name localhost;
        client_max_body_size 16m;
        recursive_error_pages on;
        error_page 413 = @too_large;
        error_page 429 = @too_many;
        error_page 418 = @writes;
        location /static/ {{
            alias "{static}/";
            disable_symlinks on;
            add_header Cache-Control "public, max-age=300";
        }}
        location ~ ^/api/v1/prediction-jobs/[0-9a-f-]+/events$ {{
            if ($request_method !~ ^(GET|HEAD)$) {{ return 418; }}
            limit_conn flight_sse 8;
            limit_conn_status 429;
            proxy_buffering off;
            proxy_cache off;
            gzip off;
            proxy_read_timeout 90s;
            proxy_pass http://flight_web;
        }}
        location / {{
            if ($request_method !~ ^(GET|HEAD)$) {{ return 418; }}
            proxy_pass http://flight_web;
        }}
        location @writes {{
            recursive_error_pages on;
            proxy_next_upstream off;
            proxy_request_buffering on;
            proxy_pass http://flight_web;
        }}
        location @too_large {{
            recursive_error_pages off;
            default_type application/json;
            return 413 '{{"success":false,"error_code":"REQUEST_TOO_LARGE","error":"request exceeds 16 MiB"}}';
        }}
        location @too_many {{
            recursive_error_pages off;
            default_type application/json;
            return 429 '{{"success":false,"error_code":"SSE_CONNECTION_LIMIT","error":"too many event connections"}}';
        }}
    }}
}}
'''


def main():
    parser = argparse.ArgumentParser(description='Render controlled two-instance Nginx configuration')
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--listen-port', type=int, default=8080)
    parser.add_argument('--web-a-port', type=int, default=7101)
    parser.add_argument('--web-b-port', type=int, default=7102)
    args = parser.parse_args()
    config = render_config(args.project_root, args.runtime_root, args.listen_port, args.web_a_port, args.web_b_port)
    runtime = safe_path(args.runtime_root)
    for name in ('body', 'proxy', 'fastcgi', 'uwsgi', 'scgi'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    output = runtime / 'nginx.conf'
    if output.is_symlink():
        raise ValueError('拒绝覆盖符号链接配置')
    output.write_text(config)
    print(output)


if __name__ == '__main__':
    main()
