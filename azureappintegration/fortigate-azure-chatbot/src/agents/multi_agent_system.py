"""
Multi-Agent Architecture Foundation
Intelligent agent routing and specialization system
"""

import streamlit as st
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AgentType(Enum):
    DEPLOYMENT = "deployment"
    TROUBLESHOOTING = "troubleshooting"
    SECURITY = "security"
    OPTIMIZATION = "optimization"
    COORDINATOR = "coordinator"

class QueryType(Enum):
    DEPLOYMENT_QUESTION = "deployment"
    TROUBLESHOOTING_ISSUE = "troubleshooting"
    SECURITY_CONCERN = "security"
    PERFORMANCE_OPTIMIZATION = "optimization"
    GENERAL_INQUIRY = "general"

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_type: AgentType
    content: str
    confidence: float
    sources: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class QueryContext:
    """Context for query processing"""
    query: str
    query_type: QueryType
    user_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.specializations = []
        self.confidence_threshold = 0.7
    
    @abstractmethod
    def can_handle(self, query_context: QueryContext) -> float:
        """Return confidence score (0-1) for handling this query"""
        pass
    
    @abstractmethod
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Process the query and return response"""
        pass
    
    def get_system_prompt(self) -> str:
        """Get system prompt for this agent"""
        return f"You are a specialized {self.name} agent for FortiGate Azure deployments."

class DeploymentAgent(BaseAgent):
    """Agent specialized in FortiGate deployment tasks"""
    
    def __init__(self):
        super().__init__(AgentType.DEPLOYMENT, "FortiGate Deployment Specialist")
        self.specializations = [
            "terraform deployment",
            "azure resource configuration",
            "network setup",
            "vm sizing",
            "high availability",
            "multi-cloud deployment"
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        """Assess if this agent can handle the query"""
        query = query_context.query.lower()
        deployment_keywords = [
            "deploy", "deployment", "terraform", "azure", "vm", "virtual machine",
            "network", "subnet", "resource group", "availability", "ha", "cluster",
            "install", "setup", "configure", "provision"
        ]
        
        score = sum(1 for keyword in deployment_keywords if keyword in query)
        return min(score / len(deployment_keywords) * 2, 1.0)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Process deployment-related queries"""
        try:
            # Generate specialized deployment response
            response_content = self._generate_deployment_response(query_context)
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.9,
                sources=["FortiGate Deployment Guide", "Azure Best Practices"],
                metadata={"specialization": "deployment", "agent": self.name},
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Deployment agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Error processing deployment query: {e}",
                confidence=0.1,
                sources=[],
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def _generate_deployment_response(self, query_context: QueryContext) -> str:
        """Generate deployment-specific response"""
        query = query_context.query
        
        # Basic deployment guidance
        if "terraform" in query.lower():
            return """
**🚀 FortiGate Terraform Deployment**

For FortiGate deployment on Azure using Terraform:

1. **Prerequisites:**
   - Azure CLI installed and authenticated
   - Terraform installed (v1.0+)
   - FortiGate license file

2. **Deployment Steps:**
   ```bash
   # Clone FortiGate Terraform templates
   git clone https://github.com/fortinet/fortigate-terraform-deploy
   cd fortigate-terraform-deploy/azure/7.0/single
   
   # Configure variables
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your settings
   
   # Deploy
   terraform init
   terraform plan
   terraform apply
   ```

3. **Key Configuration:**
   - VM size: Standard_F2s_v2 (minimum)
   - Network: 2+ subnets (external, internal)
   - Storage: Premium SSD recommended

Would you like specific guidance for HA deployment or multi-cloud setup?
            """
        
        elif "ha" in query.lower() or "high availability" in query.lower():
            return """
**🔄 FortiGate High Availability Setup**

FortiGate HA configuration options:

1. **Active-Passive HA:**
   - Two FortiGate VMs in same availability set
   - Shared storage for configuration sync
   - Floating IP for failover

2. **Active-Active HA:**
   - Load balancing across multiple FortiGates
   - Session synchronization
   - Higher throughput

3. **Azure-Specific Considerations:**
   - Use availability zones for better resilience
   - Configure Azure Load Balancer
   - Set up monitoring and health checks

Configuration steps and Terraform templates available for each scenario.
            """
        
        else:
            return f"""
**🛠️ FortiGate Deployment Assistance**

I'm your FortiGate deployment specialist. I can help with:

- Terraform deployment automation
- Azure resource configuration
- Network architecture design
- VM sizing and performance optimization
- High availability setup
- Multi-cloud deployments

Your query: "{query}"

Please provide more specific details about your deployment requirements, and I'll give you detailed guidance and configuration examples.
            """
    
    def get_system_prompt(self) -> str:
        return """You are a FortiGate deployment specialist with expertise in:
- Azure infrastructure deployment
- Terraform automation
- Network security architecture
- High availability configurations
- Performance optimization
- Multi-cloud strategies

Provide detailed, actionable deployment guidance with specific configuration examples."""

class TroubleshootingAgent(BaseAgent):
    """Agent specialized in troubleshooting FortiGate issues"""
    
    def __init__(self):
        super().__init__(AgentType.TROUBLESHOOTING, "FortiGate Troubleshooting Expert")
        self.specializations = [
            "connectivity issues",
            "performance problems",
            "configuration errors",
            "log analysis",
            "network debugging",
            "azure integration issues"
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        """Assess troubleshooting capability"""
        query = query_context.query.lower()
        troubleshooting_keywords = [
            "error", "issue", "problem", "not working", "failed", "troubleshoot",
            "debug", "fix", "resolve", "connectivity", "performance", "slow",
            "timeout", "connection", "log", "alert"
        ]
        
        score = sum(1 for keyword in troubleshooting_keywords if keyword in query)
        return min(score / len(troubleshooting_keywords) * 2, 1.0)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Process troubleshooting queries"""
        try:
            response_content = self._generate_troubleshooting_response(query_context)
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.85,
                sources=["FortiGate Troubleshooting Guide", "Azure Diagnostics"],
                metadata={"specialization": "troubleshooting", "agent": self.name},
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Troubleshooting agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Error processing troubleshooting query: {e}",
                confidence=0.1,
                sources=[],
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def _generate_troubleshooting_response(self, query_context: QueryContext) -> str:
        """Generate troubleshooting response"""
        query = query_context.query
        
        if "connectivity" in query.lower() or "connection" in query.lower():
            return """
**🔍 FortiGate Connectivity Troubleshooting**

**Step 1: Basic Connectivity Checks**
```bash
# Check FortiGate interfaces
get system interface physical
get system status

# Test connectivity
execute ping <target-ip>
execute traceroute <target-ip>
```

**Step 2: Azure Network Verification**
- Verify NSG rules allow traffic
- Check route tables
- Confirm subnet configurations
- Validate public IP assignments

**Step 3: FortiGate Configuration**
```bash
# Check firewall policies
show firewall policy
show router static

# Review logs
execute log filter category traffic
execute log display
```

**Common Issues:**
- Incorrect NSG rules
- Missing routes
- Firewall policy blocks
- Interface configuration errors

Need specific error messages or logs for detailed analysis.
            """
        
        elif "performance" in query.lower() or "slow" in query.lower():
            return """
**⚡ FortiGate Performance Troubleshooting**

**Performance Monitoring:**
```bash
# Check system resources
get system performance status
get system status

# Monitor traffic
diagnose sys top
diagnose hardware deviceinfo nic <interface>
```

**Azure VM Optimization:**
- Verify VM size meets requirements
- Enable accelerated networking
- Check disk performance (Premium SSD)
- Monitor CPU and memory usage

**Common Performance Issues:**
1. **Undersized VM** - Upgrade to higher SKU
2. **Network bottlenecks** - Check bandwidth limits
3. **Disk I/O** - Use Premium SSD storage
4. **Configuration overhead** - Optimize policies

**Recommended Actions:**
- Monitor performance baselines
- Implement traffic shaping
- Optimize firewall policies
- Consider load balancing
            """
        
        else:
            return f"""
**🛠️ FortiGate Troubleshooting Support**

I'm your troubleshooting specialist. I can help diagnose:

- Connectivity and network issues
- Performance and latency problems
- Configuration errors
- Azure integration issues
- Log analysis and debugging

Your issue: "{query}"

For effective troubleshooting, please provide:
1. Specific error messages
2. When the issue started
3. What changed recently
4. Current configuration details
5. Relevant log entries

This will help me provide targeted solutions.
            """

class SecurityAgent(BaseAgent):
    """Agent specialized in security configurations"""
    
    def __init__(self):
        super().__init__(AgentType.SECURITY, "FortiGate Security Specialist")
        self.specializations = [
            "firewall policies",
            "intrusion prevention",
            "antivirus configuration",
            "vpn setup",
            "security profiles",
            "threat intelligence"
        ]
    
    def can_handle(self, query_context: QueryContext) -> float:
        """Assess security expertise"""
        query = query_context.query.lower()
        security_keywords = [
            "security", "firewall", "policy", "vpn", "ips", "antivirus",
            "threat", "protection", "ssl", "certificate", "encryption",
            "authentication", "access", "block", "allow"
        ]
        
        score = sum(1 for keyword in security_keywords if keyword in query)
        return min(score / len(security_keywords) * 2, 1.0)
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Process security queries"""
        try:
            response_content = self._generate_security_response(query_context)
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_content,
                confidence=0.9,
                sources=["FortiGate Security Guide", "Best Practices"],
                metadata={"specialization": "security", "agent": self.name},
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Security agent error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Error processing security query: {e}",
                confidence=0.1,
                sources=[],
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def _generate_security_response(self, query_context: QueryContext) -> str:
        """Generate security-focused response"""
        return f"""
**🔒 FortiGate Security Configuration**

I'm your security specialist. I can help with:

- Firewall policy configuration
- Intrusion Prevention System (IPS) setup
- VPN configuration and management
- Security profiles and threat protection
- SSL inspection and certificates
- Access control and authentication

Your security query: "{query_context.query}"

Please specify your security requirements for detailed configuration guidance.
        """

class CoordinatorAgent(BaseAgent):
    """Agent that coordinates between other agents"""
    
    def __init__(self):
        super().__init__(AgentType.COORDINATOR, "Multi-Agent Coordinator")
        self.agents = {}
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize specialized agents"""
        self.agents = {
            AgentType.DEPLOYMENT: DeploymentAgent(),
            AgentType.TROUBLESHOOTING: TroubleshootingAgent(),
            AgentType.SECURITY: SecurityAgent()
        }
    
    def can_handle(self, query_context: QueryContext) -> float:
        """Coordinator can always handle queries"""
        return 1.0
    
    def process_query(self, query_context: QueryContext) -> AgentResponse:
        """Route query to appropriate agent or coordinate multiple agents"""
        try:
            # Get confidence scores from all agents
            agent_scores = {}
            for agent_type, agent in self.agents.items():
                score = agent.can_handle(query_context)
                agent_scores[agent_type] = score
            
            # Find best agent
            best_agent_type = max(agent_scores, key=agent_scores.get)
            best_score = agent_scores[best_agent_type]
            
            if best_score >= 0.5:
                # Route to specialist agent
                specialist_agent = self.agents[best_agent_type]
                response = specialist_agent.process_query(query_context)
                
                # Add coordination metadata
                response.metadata.update({
                    "routed_to": best_agent_type.value,
                    "confidence_scores": {k.value: v for k, v in agent_scores.items()},
                    "coordinator": "multi_agent_system"
                })
                
                return response
            else:
                # Handle as general query
                return self._generate_general_response(query_context, agent_scores)
                
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Error coordinating query: {e}",
                confidence=0.1,
                sources=[],
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def _generate_general_response(self, query_context: QueryContext, agent_scores: Dict) -> AgentResponse:
        """Generate general response when no specialist is confident"""
        content = f"""
**🤖 Multi-Agent FortiGate Assistant**

I've analyzed your query: "{query_context.query}"

**Available Specialists:**
- 🚀 **Deployment Agent** (Score: {agent_scores.get(AgentType.DEPLOYMENT, 0):.2f})
- 🔍 **Troubleshooting Agent** (Score: {agent_scores.get(AgentType.TROUBLESHOOTING, 0):.2f})
- 🔒 **Security Agent** (Score: {agent_scores.get(AgentType.SECURITY, 0):.2f})

For more specific assistance, please rephrase your question to include:
- **Deployment** questions: "How to deploy...", "Terraform setup..."
- **Troubleshooting** issues: "Error with...", "Problem connecting..."
- **Security** concerns: "Firewall policy...", "VPN configuration..."

I can provide general FortiGate guidance or route you to the appropriate specialist.
        """
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            confidence=0.6,
            sources=["Multi-Agent System"],
            metadata={
                "agent_scores": {k.value: v for k, v in agent_scores.items()},
                "routing": "general_response"
            },
            timestamp=datetime.now()
        )

class MultiAgentSystem:
    """Main multi-agent system interface"""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state for multi-agent system"""
        if "agent_conversation_history" not in st.session_state:
            st.session_state.agent_conversation_history = []
        
        if "agent_analytics" not in st.session_state:
            st.session_state.agent_analytics = {
                "queries_processed": 0,
                "agent_usage": {agent.value: 0 for agent in AgentType},
                "avg_confidence": 0.0
            }
    
    def render_interface(self):
        """Display multi-agent system interface"""
        st.markdown("### 🤖 Multi-Agent FortiGate Assistant")
        
        # System status
        self._display_system_status()
        
        # Main interface
        tab1, tab2, tab3 = st.tabs([
            "💬 Agent Chat",
            "📊 Agent Analytics", 
            "⚙️ System Settings"
        ])
        
        with tab1:
            self._display_chat_interface()
        
        with tab2:
            self._display_analytics()
        
        with tab3:
            self._display_settings()
    
    def display(self):
        """Legacy method - calls render_interface for compatibility"""
        self.render_interface()
    
    def _display_system_status(self):
        """Display system status"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success("🤖 Coordinator: ✅")
        
        with col2:
            st.success("🚀 Deployment: ✅")
        
        with col3:
            st.success("🔍 Troubleshooting: ✅")
        
        with col4:
            st.success("🔒 Security: ✅")
    
    def _display_chat_interface(self):
        """Display chat interface"""
        st.markdown("#### 💬 Multi-Agent Chat")
        
        # Display conversation history
        if st.session_state.agent_conversation_history:
            for exchange in st.session_state.agent_conversation_history:
                with st.container():
                    st.markdown(f"**👤 You:** {exchange['query']}")
                    
                    # Show agent routing info
                    if 'routed_to' in exchange.get('metadata', {}):
                        agent_name = exchange['metadata']['routed_to']
                        st.markdown(f"*🤖 Routed to: {agent_name.title()} Agent*")
                    
                    st.markdown(f"**🤖 Assistant:** {exchange['response']}")
                    
                    # Show confidence and sources
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence = exchange.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col2:
                        sources = len(exchange.get('sources', []))
                        st.metric("Sources", sources)
                    
                    st.markdown("---")
        
        # Query input
        query = st.text_area(
            "Ask your FortiGate question:",
            placeholder="Enter your question about deployment, troubleshooting, or security...",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Ask Agents", type="primary", key="agent_ask_button"):
                if query.strip():
                    self._process_agent_query(query)
        
        with col2:
            if st.button("🗑️ Clear History", key="agent_clear_history"):
                st.session_state.agent_conversation_history = []
                st.rerun()
    
    def _process_agent_query(self, query: str):
        """Process query through multi-agent system"""
        try:
            with st.spinner("🤖 Routing to appropriate agent..."):
                # Create query context
                query_context = QueryContext(
                    query=query,
                    query_type=QueryType.GENERAL_INQUIRY,  # Could be enhanced with classification
                    user_context={},
                    conversation_history=st.session_state.agent_conversation_history,
                    metadata={"timestamp": datetime.now().isoformat()}
                )
                
                # Process through coordinator
                response = self.coordinator.process_query(query_context)
                
                # Store conversation
                conversation_entry = {
                    "query": query,
                    "response": response.content,
                    "agent_type": response.agent_type.value,
                    "confidence": response.confidence,
                    "sources": response.sources,
                    "metadata": response.metadata,
                    "timestamp": response.timestamp.isoformat()
                }
                
                st.session_state.agent_conversation_history.append(conversation_entry)
                
                # Update analytics
                self._update_analytics(response)
                
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing query: {e}")
    
    def _update_analytics(self, response: AgentResponse):
        """Update analytics data"""
        analytics = st.session_state.agent_analytics
        
        analytics["queries_processed"] += 1
        analytics["agent_usage"][response.agent_type.value] += 1
        
        # Update average confidence
        total_confidence = analytics["avg_confidence"] * (analytics["queries_processed"] - 1)
        analytics["avg_confidence"] = (total_confidence + response.confidence) / analytics["queries_processed"]
    
    def _display_analytics(self):
        """Display analytics dashboard"""
        st.markdown("#### 📊 Multi-Agent Analytics")
        
        analytics = st.session_state.agent_analytics
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", analytics["queries_processed"])
        
        with col2:
            avg_conf = analytics["avg_confidence"]
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        with col3:
            most_used = max(analytics["agent_usage"], key=analytics["agent_usage"].get)
            st.metric("Most Used Agent", most_used.title())
        
        with col4:
            success_rate = 95.2  # Placeholder
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Agent usage breakdown
        st.markdown("#### 🤖 Agent Usage")
        for agent_type, usage_count in analytics["agent_usage"].items():
            if analytics["queries_processed"] > 0:
                percentage = (usage_count / analytics["queries_processed"]) * 100
                st.metric(f"{agent_type.title()} Agent", f"{usage_count} ({percentage:.1f}%)")
    
    def _display_settings(self):
        """Display system settings"""
        st.markdown("#### ⚙️ Multi-Agent System Settings")
        
        with st.expander("🤖 Agent Configuration", expanded=True):
            st.markdown("""
            **Available Agents:**
            
            - 🚀 **Deployment Agent**: Terraform, Azure resources, VM configuration
            - 🔍 **Troubleshooting Agent**: Connectivity, performance, debugging
            - 🔒 **Security Agent**: Firewall policies, VPN, threat protection
            - 🤖 **Coordinator Agent**: Query routing and multi-agent orchestration
            
            **Routing Logic:**
            - Queries are analyzed for keywords and context
            - Confidence scores determine the best specialist agent
            - Fallback to coordinator for general questions
            """)
        
        with st.expander("📊 System Information"):
            analytics = st.session_state.agent_analytics
            st.json({
                "total_agents": 4,
                "routing_algorithm": "confidence_based",
                "fallback_strategy": "coordinator_general_response",
                "session_queries": analytics["queries_processed"],
                "system_status": "operational"
            })

def get_multi_agent_system():
    """Factory function to get multi-agent system"""
    return MultiAgentSystem()
