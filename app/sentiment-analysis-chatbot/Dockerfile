# Stage 1: Build the Angular application
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the application
# We use the production configuration
RUN npm run build --configuration=production

# Consolidate build outputs just in case it uses the application builder (Angular 17+)
# which outputs to dist/sentiment-analysis-chatbot/browser
RUN if [ -d "dist/sentiment-analysis-chatbot/browser" ]; then mv dist/sentiment-analysis-chatbot/browser /app/browser_temp && rm -rf dist/sentiment-analysis-chatbot && mv /app/browser_temp dist/sentiment-analysis-chatbot; fi

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Copy the Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built application from the build stage
COPY --from=build /app/dist/sentiment-analysis-chatbot /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
