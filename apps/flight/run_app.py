"""Start the isolated flight Web app without importing model runtimes."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'generated')]
from apps.flight.app import create_app

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=5000)
    args=parser.parse_args()
    app=create_app()
    try: app.run(host=args.host,port=args.port,debug=False,threaded=True)
    finally: app.extensions['prediction_client'].close()

if __name__=='__main__': main()
