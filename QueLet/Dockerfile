# Use Maven image to build the application
FROM maven:3.8.5-openjdk-17 AS build

# Set working directory in the container
WORKDIR /QueLet

# Copy the pom.xml and download dependencies
COPY /QueLet/pom.xml .
RUN mvn dependency:go-offline -B

# Copy the entire project and build it
COPY ./QueLet .
RUN mvn clean package -DskipTests

# Use a smaller JDK image for the runtime environment
FROM openjdk:17-jdk-slim

# Set working directory
WORKDIR /QueLet

# Copy the built JAR file from the build stage
COPY --from=build /QueLet/target/spring-boot-docker.jar /QueLet/spring-boot-docker.jar

# Expose the application port (if using a different port, update accordingly)
EXPOSE 8080

# Define the command to run the application
ENTRYPOINT ["java", "-jar", "spring-boot-docker.jar"]
