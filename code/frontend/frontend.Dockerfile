FROM node:14-alpine as webpack-built
WORKDIR /app

COPY package.json .
RUN yarn install --silent

COPY src src
COPY public public
RUN yarn run build

FROM nginx:alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY robots.txt /web/robots.txt
COPY --from=webpack-built /app/build /web/app

VOLUME /web/static/ /web/media/
EXPOSE 80 443
