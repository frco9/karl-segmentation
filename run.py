import os

from api.app import create_app

app = create_app()

if __name__ == '__main__':
  port = os.getenv('PORT')
  # run app
  app.run(host='0.0.0.0', port=port)