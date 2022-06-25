FROM node:14-alpine AS webpack-built
WORKDIR /app
COPY package.json .
RUN yarn install --silent
COPY src src
COPY public public
RUN yarn run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx.run.sh /etc/nginx.run.sh
COPY demosecre[t]s /etc/demosecrets
COPY --from=webpack-built /app/build /web/app
VOLUME /web/static/ /web/media/
EXPOSE 80 443
CMD ["/etc/nginx.run.sh"]
