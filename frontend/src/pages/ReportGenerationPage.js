import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  TextField, 
  Button, 
  Box, 
  Paper, 
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Divider,
  FormControlLabel,
  Switch,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  InputAdornment,
  Tooltip,
  Autocomplete,
  Chip,
  Snackbar
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import InfoIcon from '@mui/icons-material/Info';
import SearchIcon from '@mui/icons-material/Search';
import { generateReport, pingAPI, listModels } from '../utils/api';
import HistoryPanel from '../components/HistoryPanel';

// Recommended model options
const commonModels = [
  { label: 'Claude 3.7 Sonnet (Recommended)', value: 'claude-3-7-sonnet-20250219' },
  { label: 'Claude 3.5 Sonnet', value: 'claude-3-5-sonnet-20241022' },
  { label: 'GPT-4o', value: 'gpt-4o' },
  { label: 'DeepSeek Chat', value: 'deepseek-chat' },
  { label: 'DeepSeek Reasoner', value: 'deepseek-reasoner' },
  { label: 'Gemini 2.5 Pro Exp', value: 'gemini-2.5-pro-exp-03-25' },
  { label: 'Gemini 2.5 Pro Preview', value: 'gemini-2.5-pro-preview-03-25' },
];

// Example prompts for report generation
const examplePrompts = [
  "What is the commercial value of a long-article writing AI Agent? Write a detailed analysis report.",
  "Write a comprehensive report on the impact of artificial intelligence on healthcare, focusing on diagnosis, treatment planning, and patient outcomes.",
  "Prepare a detailed report on sustainable energy solutions for developing countries, including their economic viability and environmental impact."
];

const ReportGenerationPage = () => {
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('claude-3-5-sonnet-20241022');
  const [llmBackend, setLlmBackend] = useState(localStorage.getItem('llm_backend') || 'auto');
  const [systemPrompt, setSystemPrompt] = useState(localStorage.getItem('system_prompt') || '');
  const [searchEngine, setSearchEngine] = useState('google');
  const [enableSearch, setEnableSearch] = useState(true);
  const [apiKeys, setApiKeys] = useState({
    openai: localStorage.getItem('openai_api_key') || '',
    claude: localStorage.getItem('claude_api_key') || '',
    gemini: localStorage.getItem('gemini_api_key') || '',
    deepseek: localStorage.getItem('deepseek_api_key') || '',
    serpapi: localStorage.getItem('serpapi_api_key') || '',
    openrouter: localStorage.getItem('openrouter_api_key') || '',
    openrouterReferer: localStorage.getItem('openrouter_referer') || '',
    openrouterTitle: localStorage.getItem('openrouter_title') || '',
  });
  const [showApiSection, setShowApiSection] = useState(false);
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showClaudeKey, setShowClaudeKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [showSerpApiKey, setShowSerpApiKey] = useState(false);
  const [showOpenRouterKey, setShowOpenRouterKey] = useState(false);
  const [showDeepSeekKey, setShowDeepSeekKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState('');
  const [remoteModels, setRemoteModels] = useState([]);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [showStatus, setShowStatus] = useState(false);
  const navigate = useNavigate();
  
  // Save API keys to localStorage when they change
  useEffect(() => {
    if (apiKeys.openai) localStorage.setItem('openai_api_key', apiKeys.openai);
    if (apiKeys.claude) localStorage.setItem('claude_api_key', apiKeys.claude);
    if (apiKeys.gemini) localStorage.setItem('gemini_api_key', apiKeys.gemini);
    if (apiKeys.deepseek) localStorage.setItem('deepseek_api_key', apiKeys.deepseek);
    if (apiKeys.serpapi) localStorage.setItem('serpapi_api_key', apiKeys.serpapi);
    if (apiKeys.openrouter) localStorage.setItem('openrouter_api_key', apiKeys.openrouter);
    if (apiKeys.openrouterReferer) localStorage.setItem('openrouter_referer', apiKeys.openrouterReferer);
    if (apiKeys.openrouterTitle) localStorage.setItem('openrouter_title', apiKeys.openrouterTitle);
  }, [apiKeys]);

  useEffect(() => {
    localStorage.setItem('llm_backend', llmBackend);
  }, [llmBackend]);

  useEffect(() => {
    localStorage.setItem('system_prompt', systemPrompt);
  }, [systemPrompt]);

  const resolveBackendForModels = () => {
    const backend = (llmBackend || 'auto').toLowerCase();
    if (backend !== 'auto') return backend;

    const candidates = [];
    if (apiKeys.deepseek) candidates.push('deepseek');
    if (apiKeys.openrouter) candidates.push('openrouter');
    if (apiKeys.openai) candidates.push('openai');
    if (apiKeys.claude) candidates.push('anthropic');
    if (apiKeys.gemini) candidates.push('gemini');
    if (candidates.length === 1) return candidates[0];
    return null;
  };

  const getApiKeyForBackend = (backend) => {
    switch ((backend || '').toLowerCase()) {
      case 'deepseek':
        return apiKeys.deepseek;
      case 'openrouter':
        return apiKeys.openrouter;
      case 'openai':
        return apiKeys.openai;
      case 'anthropic':
        return apiKeys.claude;
      case 'gemini':
        return apiKeys.gemini;
      default:
        return '';
    }
  };

  const handleSearchModels = async () => {
    setModelsError('');
    const backend = resolveBackendForModels();
    if (!backend) {
      setModelsError('Select an LLM Backend (or provide exactly one API key) to search models.');
      setShowApiSection(true);
      return;
    }

    const apiKey = getApiKeyForBackend(backend);
    if (!apiKey) {
      setModelsError(`Missing API key for ${backend}.`);
      setShowApiSection(true);
      return;
    }

    setModelsLoading(true);
    try {
      const resp = await listModels({
        backend,
        apiKey,
        query: model || '',
        limit: 50
      });
      setRemoteModels(resp.models || []);
      setShowStatus(true);
      setStatusMessage(`Found ${resp.count || (resp.models || []).length} models for ${backend}.`);
    } catch (err) {
      setModelsError(err.message || 'Failed to list models');
    } finally {
      setModelsLoading(false);
    }
  };

  const modelOptions = React.useMemo(() => {
    const byId = new Map();
    for (const m of commonModels) byId.set(m.value, m);
    for (const id of remoteModels || []) {
      if (!id) continue;
      if (!byId.has(id)) byId.set(id, { label: id, value: id });
    }
    return Array.from(byId.values());
  }, [remoteModels]);
  
  const handleApiKeyChange = (provider, value) => {
    setApiKeys(prev => ({
      ...prev,
      [provider]: value
    }));
  };

  // Check if API is available on component mount
  useEffect(() => {
    async function checkAPIConnection() {
      try {
        await pingAPI();
        // API is available, nothing to do
      } catch (err) {
        setError('Cannot connect to the backend server. Please make sure it is running at http://localhost:' + (process.env.REACT_APP_BACKEND_PORT || '5001') + '.');
      }
    }
    
    checkAPIConnection();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!prompt) {
      setError('Please provide a prompt for report generation.');
      return;
    }
    
    // Check if the appropriate API keys are provided
    const modelLower = (model || '').toLowerCase();
    const backend = (llmBackend || 'auto').toLowerCase();
    let effectiveBackend = backend;
    
    if (backend === 'auto') {
      if (modelLower.startsWith('openrouter:') || modelLower.includes('/')) effectiveBackend = 'openrouter';
      else if (modelLower.includes('claude')) effectiveBackend = 'anthropic';
      else if (modelLower.includes('gemini')) effectiveBackend = 'gemini';
      else if (modelLower.includes('deepseek')) effectiveBackend = 'deepseek';
      else effectiveBackend = 'openai';
    }
    
    if (effectiveBackend === 'openrouter' && !apiKeys.openrouter) {
      setError('Please provide your OpenRouter API key in the settings section.');
      setShowApiSection(true);
      return;
    }
    
    if (effectiveBackend === 'openai' && !apiKeys.openai) {
      setError('Please provide your OpenAI API key in the settings section.');
      setShowApiSection(true);
      return;
    }
    
    if (effectiveBackend === 'anthropic' && !apiKeys.claude) {
      setError('Please provide your Anthropic API key in the settings section.');
      setShowApiSection(true);
      return;
    }
    
    if (effectiveBackend === 'gemini' && !apiKeys.gemini) {
      setError('Please provide your Google Gemini API key in the settings section.');
      setShowApiSection(true);
      return;
    }
    
    if (effectiveBackend === 'deepseek' && !apiKeys.deepseek) {
      setError('Please provide your DeepSeek API key in the settings section.');
      setShowApiSection(true);
      return;
    }
    
    if (enableSearch && !apiKeys.serpapi) {
      setError('Please provide your SerpAPI key in the settings section to enable search functionality.');
      setShowApiSection(true);
      return;
    }
    
    // First, check if the server is reachable
    try {
      await pingAPI();
    } catch (err) {
      setError('Cannot connect to the backend server. Please make sure it is running at http://localhost:' + (process.env.REACT_APP_BACKEND_PORT || '5001') + '.');
      return;
    }
    
    setLoading(true);
    setError('');
    setStatusMessage('Initiating report generation...');
    setShowStatus(true);
    
    try {
      // Call the backend API to start report generation
      const response = await generateReport({
        prompt,
        model,
        llmBackend,
        systemPrompt,
        enableSearch,
        searchEngine,
        apiKeys: {
          openai: apiKeys.openai,
          claude: apiKeys.claude,
          gemini: apiKeys.gemini,
          deepseek: apiKeys.deepseek,
          serpapi: apiKeys.serpapi,
          openrouter: apiKeys.openrouter,
          openrouterReferer: apiKeys.openrouterReferer,
          openrouterTitle: apiKeys.openrouterTitle,
        }
      });
      
      // Navigate to the results page with the task ID
      if (response && response.taskId) {
        setStatusMessage('Report generation started successfully!');
        navigate(`/results/${response.taskId}`, { 
          state: { 
            taskId: response.taskId,
            prompt,
            model,
            searchEngine: enableSearch ? searchEngine : 'none',
            type: 'report',
            status: 'generating'
          } 
        });
      } else {
        throw new Error('No task ID returned from the server');
      }
    } catch (err) {
      setLoading(false);
      setStatusMessage('');
      setError('Error starting report generation: ' + (err.message || 'Unknown error'));
      console.error('Report generation error:', err);
    }
  };

  const handleExampleClick = (example) => {
    setPrompt(example);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 6 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Technical Report Generation
        </Typography>
        <Typography variant="body1" paragraph>
          Generate comprehensive technical reports using our Heterogeneous Recursive Planning framework.
          The system integrates information retrieval, logical reasoning, and content composition to 
          create well-structured and informative reports.
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <HistoryPanel />
      
      <Snackbar
        open={showStatus}
        autoHideDuration={6000}
        onClose={() => setShowStatus(false)}
        message={statusMessage}
      />

      <Paper elevation={3} sx={{ p: 4, mb: 6 }}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                label="Report Topic"
                multiline
                rows={6}
                fullWidth
                required
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the technical report you want to generate..."
                variant="outlined"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <Autocomplete
                freeSolo
                options={modelOptions}
                getOptionLabel={(option) => {
                  if (typeof option === 'string') {
                    return option;
                  }
                  return option.label || '';
                }}
                value={model}
                onChange={(event, newValue) => {
                  if (typeof newValue === 'string') {
                    setModel(newValue);
                  } else if (newValue && newValue.value) {
                    setModel(newValue.value);
                  } else {
                    setModel('');
                  }
                }}
                onInputChange={(event, newInputValue) => {
                  if (event && event.type === 'change') {
                    setModel(newInputValue);
                  }
                }}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Model"
                    variant="outlined"
                    fullWidth
                    placeholder="Enter or select a model"
                    helperText={modelsError ? `Model search error: ${modelsError}` : (modelsLoading ? 'Searching models...' : 'Enter any model name. For OpenRouter, use provider/model (e.g. openai/gpt-4o).')}
                    InputProps={{
                      ...params.InputProps,
                      endAdornment: (
                        <>
                          <Tooltip title="Search available models for your selected backend">
                            <span>
                              <IconButton
                                size="small"
                                onClick={handleSearchModels}
                                disabled={modelsLoading}
                                edge="end"
                              >
                                <SearchIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                          {params.InputProps.endAdornment}
                        </>
                      ),
                    }}
                  />
                )}
                renderOption={(props, option) => (
                  <li {...props}>
                    <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="body1">{option.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {option.value}
                      </Typography>
                    </Box>
                  </li>
                )}
                renderTags={(value, getTagProps) => 
                  value.map((option, index) => (
                    <Chip
                      label={option.label}
                      size="small"
                      {...getTagProps({ index })}
                    />
                  ))
                }
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch 
                    checked={enableSearch} 
                    onChange={(e) => setEnableSearch(e.target.checked)} 
                  />
                }
                label="Enable Search"
              />
              
              <FormControl fullWidth sx={{ mt: 1 }} disabled={!enableSearch}>
                <InputLabel id="search-engine-label">Search Engine</InputLabel>
                <Select
                  labelId="search-engine-label"
                  id="search-engine-select"
                  value={searchEngine}
                  label="Search Engine"
                  onChange={(e) => setSearchEngine(e.target.value)}
                >
                  <MenuItem value="google">Google</MenuItem>
                  <MenuItem value="bing">Bing</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={4} sx={{ display: 'flex', alignItems: 'center' }}>
              <Button
                type="submit"
                variant="contained"
                color="secondary"
                size="large"
                fullWidth
                disabled={loading || !prompt}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Generate Report'}
              </Button>
            </Grid>
            
            <Grid item xs={12}>
              <Accordion 
                expanded={showApiSection}
                onChange={() => setShowApiSection(!showApiSection)}
                sx={{
                  mt: 2,
                  backgroundColor: 'grey.50',
                  boxShadow: 'none',
                  '&:before': {
                    display: 'none',
                  },
                  border: '1px solid',
                  borderColor: 'grey.200',
                  borderRadius: '8px !important',
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="api-keys-content"
                  id="api-keys-header"
                  sx={{ borderRadius: 2 }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                    API Settings
                    <Tooltip title="Your API keys are stored locally in your browser and sent only to your backend server for this app">
                      <IconButton size="small" sx={{ ml: 1 }}>
                        <InfoIcon fontSize="small" color="action" />
                      </IconButton>
                    </Tooltip>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Grid container spacing={3}>
                        <Grid item xs={12}>
                          <FormControl fullWidth>
                            <InputLabel id="llm-backend-label">LLM Backend</InputLabel>
                            <Select
                              labelId="llm-backend-label"
                              value={llmBackend}
                              label="LLM Backend"
                              onChange={(e) => setLlmBackend(e.target.value)}
                            >
                              <MenuItem value="auto">Auto</MenuItem>
                              <MenuItem value="openai">OpenAI</MenuItem>
                              <MenuItem value="anthropic">Anthropic</MenuItem>
                              <MenuItem value="gemini">Gemini</MenuItem>
                              <MenuItem value="openrouter">OpenRouter</MenuItem>
                              <MenuItem value="deepseek">DeepSeek</MenuItem>
                            </Select>
                          </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="System Prompt (optional)"
                            fullWidth
                            variant="outlined"
                            value={systemPrompt}
                            onChange={(e) => setSystemPrompt(e.target.value)}
                            multiline
                            minRows={3}
                            placeholder="Add global instructions for the model..."
                            helperText="Prepended to every system message in this run"
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="OpenAI API Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.openai}
                            onChange={(e) => handleApiKeyChange('openai', e.target.value)}
                            type={showOpenAIKey ? 'text' : 'password'}
                            placeholder="sk-..."
                            helperText="Required for GPT models"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                                    edge="end"
                                  >
                                    {showOpenAIKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="Anthropic API Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.claude}
                            onChange={(e) => handleApiKeyChange('claude', e.target.value)}
                            type={showClaudeKey ? 'text' : 'password'}
                            placeholder="sk-ant-..."
                            helperText="Required for Claude models"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowClaudeKey(!showClaudeKey)}
                                    edge="end"
                                  >
                                    {showClaudeKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="OpenRouter API Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.openrouter}
                            onChange={(e) => handleApiKeyChange('openrouter', e.target.value)}
                            type={showOpenRouterKey ? 'text' : 'password'}
                            placeholder="sk-or-..."
                            helperText="Required for OpenRouter models (e.g. openai/gpt-4o, anthropic/claude-3.5-sonnet)"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowOpenRouterKey(!showOpenRouterKey)}
                                    edge="end"
                                  >
                                    {showOpenRouterKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="DeepSeek API Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.deepseek}
                            onChange={(e) => handleApiKeyChange('deepseek', e.target.value)}
                            type={showDeepSeekKey ? 'text' : 'password'}
                            placeholder="sk-..."
                            helperText="Required for DeepSeek models (e.g. deepseek-chat)"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowDeepSeekKey(!showDeepSeekKey)}
                                    edge="end"
                                  >
                                    {showDeepSeekKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Grid container spacing={3}>
                        <Grid item xs={12}>
                          <TextField
                            label="Google Gemini API Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.gemini}
                            onChange={(e) => handleApiKeyChange('gemini', e.target.value)}
                            type={showGeminiKey ? 'text' : 'password'}
                            placeholder="your-api-key-..."
                            helperText="Required for Gemini models"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowGeminiKey(!showGeminiKey)}
                                    edge="end"
                                  >
                                    {showGeminiKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="SerpAPI Key"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.serpapi}
                            onChange={(e) => handleApiKeyChange('serpapi', e.target.value)}
                            type={showSerpApiKey ? 'text' : 'password'}
                            placeholder="..."
                            helperText="Required for search functionality"
                            InputProps={{
                              endAdornment: (
                                <InputAdornment position="end">
                                  <IconButton
                                    aria-label="toggle password visibility"
                                    onClick={() => setShowSerpApiKey(!showSerpApiKey)}
                                    edge="end"
                                  >
                                    {showSerpApiKey ? <VisibilityOff /> : <Visibility />}
                                  </IconButton>
                                </InputAdornment>
                              ),
                            }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="OpenRouter HTTP-Referer (optional)"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.openrouterReferer}
                            onChange={(e) => handleApiKeyChange('openrouterReferer', e.target.value)}
                            placeholder="https://your.site"
                            helperText="Optional header for OpenRouter rankings"
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <TextField
                            label="OpenRouter X-Title (optional)"
                            fullWidth
                            variant="outlined"
                            value={apiKeys.openrouterTitle}
                            onChange={(e) => handleApiKeyChange('openrouterTitle', e.target.value)}
                            placeholder="WriteHERE"
                            helperText="Optional header for OpenRouter rankings"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">
                        Your API keys are stored in your browser's local storage and are sent to the backend server of this app to run the generation.
                      </Typography>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Grid>
          </Grid>
        </form>
      </Paper>

      <Box sx={{ mb: 6 }}>
        <Typography variant="h5" gutterBottom>
          Example Topics
        </Typography>
        <Typography variant="body2" paragraph>
          Click on any example to use it as your prompt:
        </Typography>
        
        <Grid container spacing={3}>
          {examplePrompts.map((example, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card 
                sx={{ 
                  height: '100%', 
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'translateY(-5px)',
                    boxShadow: 3
                  }
                }}
                onClick={() => handleExampleClick(example)}
              >
                <CardContent>
                  <Typography variant="body2" color="text.secondary">
                    {example.length > 200 ? example.substring(0, 200) + '...' : example}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      <Paper elevation={3} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h5" gutterBottom>
          Tips for Effective Report Prompts
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" fontWeight="bold">
              Define Scope
            </Typography>
            <Typography variant="body2">
              Clearly specify the scope and focus of your report to ensure the content
              addresses your specific needs.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" fontWeight="bold">
              Indicate Structure
            </Typography>
            <Typography variant="body2">
              If you have specific requirements for the structure or sections of the report,
              mention them in your prompt.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" fontWeight="bold">
              Specify Depth
            </Typography>
            <Typography variant="body2">
              Indicate whether you need a general overview or an in-depth analysis with
              detailed technical information and citations.
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default ReportGenerationPage;
