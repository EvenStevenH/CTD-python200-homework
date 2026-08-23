# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1

The core economic model of cloud computing is a pay-as-you-go or usage-based pricing model, where you only pay for the specific resources (like compute power, storage, network bandwidth) that you actually use. It's different from owning your own servers, where you must purchase and maintain hardware.

## Cloud Concepts Question 2

Vertical scaling means increasing the resources of a single machine (adding more CPU cores, RAM, or GPU power), while horizontal scaling means adding more machines to distribute the workload.

I might choose vertical scaling when I need more memory and faster GPU for training models; I can upgrade an existing VM with more CPUs, RAM, and GPU. Conversely I might choose horizontal scaling if an app I manage suddenly receives a lot of traffic after launch; I can add multiple identical servers/instances to help route incoming requests.

- Where "a web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch", I'd apply horizontal scaling; the sudden surge in users requires distributing the increased workload across additional web servers.

- Where "a data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM", I'd apply vertical scaling so existing VM resources of the single machine can be upgraded.

- Where "a data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines", I'd apply horizontal scaling to split the work across multiple identical worker machines.

## Cloud Concepts Question 3

- Gmail — SaaS: you use a fully managed application, and Microsoft handles everything from servers to updates.

- Azure Virtual Machines — IaaS: you get raw virtual machines and manage the OS, software, and configuration yourself.

- AWS S3 (Simple Storage Service) — IaaS: it provides fundamental storage infrastructure that you configure and use directly.

- GitHub Codespaces — PaaS: GitHub manages the underlying environment; you bring your code and configuration.

- Snowflake — SaaS: you interact with a fully managed analytics service without managing servers or infrastructure.

- Supabase — SaaS: a hosted database and backend-as-a-service where the provider manages the stack.

Definitions:

- IaaS (Infrastructure as a Service): the provider gives you raw computing resources (such as virtual machines, storage, networking) and you manage everything from the operating system upward. for example, Azure Virtual Machines; the developer is responsible for installing software, configuring security, and maintaining the environment.

- PaaS (Platform as a Service): the provider manages the infrastructure while you bring your own code. The platform handles running, scaling, and keeping the underlying machine healthy. For example, GitHub Codespaces; the developer is responsible for application code and configuration.

- SaaS (Software as a Service): a fully managed application that you use through a browser or client; you don't think about servers at all. For example, Google Docs; the provider manages everything, and you just log in and use it.

## Cloud Concepts Question 4

A managed data platform like Databricks or Snowflake is built on top of cloud providers, but abstracts away much of the complexity by pre-configuring and optimizing for very specific workloads (such as data processing and machine learning). In other words: instead of assembling your own stack from individual cloud providers like AWS or GCP directly, the platform manages resources on your behalf. You gain convenience through less infrastructure management and optimizations for large-scale data operations. However, you do give up some flexibility and control because your workload runs in a curated environment that may not allow fine-tuning or certain integrations.

## Cloud Concepts Question 5

Two situations where cloud computing is probably not the right choice:

- When working with small datasets that fit comfortably on local hardware, as the overhead of cloud resources can outweigh benefits

- The learning curve and cost complexity can potentially outweigh the benefits; even simple tasks in the cloud can take a while, and you have to figure out the right resources and jargon initially

<!-- ----------------------------------------------------------------------- -->

# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

- Amazon Web Services (AWS); the broadest service catalog and over a third of the cloud market, making it the default choice for large enterprises, startups, and nonprofits with engineering staff.

- Google Cloud Platform (GCP); the strongest in data analytics and machine learning, built on Google's foundational distributed systems work like MapReduce and BigQuery, and is often preferred by teams working at scale on ML infrastructure.

- Microsoft Azure; dominates enterprise and government settings due to its deep integration with Windows, Active Directory, and Microsoft 365, making it a good choice for large nonprofits and public-sector organizations already invested in Microsoft.

## Cloud Landscape Question 2

The course switched from Azure to Supabase for three reasons:

- Access — Azure requires organizational provisioning, which may delay usage; Supabase accounts are quickly self-provisioned at supabase.com with a free tier sufficient for the course.

- Pedagogical fit — Azure Blob Storage stores data as opaque files organized by path, while Supabase provides rows and columns in a relational database. This makes querying, filtering, and reasoning about structured data more transferable to other data roles.

- Pipeline coherence — the ETL pipeline maps naturally onto two tables with clear relationships in Supabase. This reinforces data model concepts and makes each stage easy to inspect and debug via direct queries.

- Reflection: when evaluating a cloud tool for a new project, I should consider not just whether the tool has the right capabilities, but also how well it fits the specific workflows I use. Especially with me being new to working with cloud data, I want to minimize complexity to focus on learning the basics/concepts (such as data model design and credential management).

## Cloud Landscape Question 3

- "Store 10 TB of image files, retrieve by filename from any machine" > Object storage > AWS S3 (or GCP Cloud Storage, Azure Blob Storage)

- "Run an ML training job on a GPU for four hours, then shut it down" > Compute > AWS EC2 (or GCP Compute Engine, Azure Virtual Machines)

- "Host a web API that scales up with traffic spikes and down when quiet" > Serverless compute > AWS Lambda (or GCP Cloud Functions, Azure Functions)

- "Send structured data to an LLM and get a text response back" > LLM API > OpenAI directly (or AWS Bedrock, GCP Vertex AI, Azure OpenAI Service)

## Cloud Landscape Question 4

A simple data project I made was weather scraper that fetches raw weather forecasts from a site. I think a stack of services I could use for it is Supabase for database storage, Cloudflare for edge compute, and AWS Bedrock for access to LLM models to create summaries.

Consolidating everything to one provider could simplify credential management, reduce latency for data movement, and potentially lower costs through bundled pricing (if available). However, you might give up the specialized strengths of each tool, like Supabase's developer-friendly relational model or BigQuery's services on GCP. Different tools excel in different domains, so consolidation is a tradeoff between simplicity and optimization.
