plugins {
    `java-library`
    `maven-publish`
}

group = "io.vietasr"
version = "0.1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
    withSourcesJar()
}

repositories {
    mavenCentral()
}

dependencies {
    api("net.java.dev.jna:jna:5.14.0")
}

sourceSets {
    main {
        java {
            srcDir("src/main/java")
        }
        resources {
            srcDir("_native")
            include("**/*.so", "**/*.dylib", "**/*.dll")
        }
    }
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            artifactId = "vietasr"
            from(components["java"])
        }
    }
}
