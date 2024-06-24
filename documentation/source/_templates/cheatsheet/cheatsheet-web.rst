{% extends 'cheatsheet-template.html' %}

{% block css %}
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.1/font/bootstrap-icons.css">
  <link rel="stylesheet" href="_static/alabaster.css">
  <link rel="stylesheet" href="_static/cheatsheet.css">
{% endblock %}

{% block jquery %}
      $("#tl").addClass("lower");
      $(".subhead.lower").css("margin-top", "0");
      $(".subhead.lower").css("margin-top", "0");
      <!-- modify some of the pdf styling -->
      $(".first-half > p").each(function() {
        $(this).html($(this).html().replace(/(&nbsp;|\s){2,}$/, ''));
      });
      <!-- Show the first section when page loads -->
      $(".content").slice(0, 3).show();
      <!-- Controls for hiding and showing content -->
      $(".heading").click(function () {
        if ($(this).hasClass("active")) {
          $(this).parent().find(".bi").removeClass("bi-caret-down-fill")
          $(this).parent().find(".bi").addClass("bi-caret-right-fill")
          $(this).parent().find(".content").slideUp(400);
          $(this).parent().find(".heading").removeClass("active");
        } else {
          $(this).parent().find(".bi").removeClass("bi-caret-right-fill")
          $(this).parent().find(".bi").addClass("bi-caret-down-fill")
          $(this).parent().find(".content").slideDown(400);
          $(this).parent().find(".heading").addClass("active");
        }
        var section = $(this).closest(".section")
        var subs = section.find(".subsection > .heading")
        var active = section.find(".subsection > .heading.active")
        var top = $(this).closest(".section").find(".heading").first();
        if (subs.length && active.length > 0) {
          top.find(".bi").removeClass("bi-caret-right-fill");
          top.find(".bi").addClass("bi-caret-down-fill");
          top.addClass("active");
          subs.parent().find(".footnote").hide();
          active.last().parent().find(".footnote").slideDown(400);
        } else if (subs.length && active.length === 0) {
          top.find(".bi").removeClass("bi-caret-down-fill");
          top.find(".bi").addClass("bi-caret-right-fill");
          top.removeClass("active");
          section.find(".footnote").hide();
        }
        return false;
      });
      $("#expand").click(function () {
        $(document).find(".heading:not(.active)").addClass("active");
        $(document).find(".bi").removeClass("bi-caret-right-fill");
        $(document).find(".bi").addClass("bi-caret-down-fill");
        $(document).find(".content").slideDown(400);
        $(document).find(".footnote").last().slideDown(400);
      });
      $("#collapse").click(function () {
        $(document).find(".heading.active").removeClass("active");
        $(document).find(".bi").removeClass("bi-caret-down-fill");
        $(document).find(".bi").addClass("bi-caret-right-fill");
        $(document).find(".content").slideUp(400);
        $(document).find(".footnote").slideUp(400);
      });
{% endblock %}

{% block style %}
  <style>
    /* Bootstrap is overriding alabaster settings for sidebar */
    div.sphinxsidebarwrapper h1, h3 {
      line-height: 1.5;
    }

    div.sphinxsidebarwrapper h1.logo {
      font-size: 2em;
      font-weight: bold;
    }

    /* Alabaster wraps everything in div class body */
    div.body tr, div.body p {
      font-size: 12px;
      font-family: "Courier New", monospace;
      margin: auto;
      line-height: 1.5;
    }

    div.body td {
      padding-right: 3rem;
    }

    div.body .section {
      margin-bottom: 0.5rem;
    }

    div.body .content {
      display: none;
      padding: 0.5rem;
    }

    div.body .footnote {
      display: none;
      padding-left: 0.5rem;
      padding-bottom: 0.5rem;
    }

    div.body .section-head {
      font-size: 16px;
    }

    div.body .subhead {
      font-size: 14px;
    }

    div.body .highlighter {
      font-size: 12px;
    }

    div.body .code {
      overflow-x: auto;
      white-space: nowrap;
      padding-left: 0.2rem;
      line-height: 1.5;
    }

    div.body .code p {
      font-size: 11px;
    }

    div.body .lower {
      margin-top: 0.5rem;
    }

    div.body .heading {
      display: inline-flex;
      width: 100%;
      border: 1px solid lightgray;
      cursor: pointer;
    }

    div.body .heading h3 {
      text-align: left;
      float: right;
    }

    div.body .first-half {
      margin-right: 0.5rem;
    }

    div.body .bi {
      padding-left: 0.2rem;
      padding-right: 0.5rem;
      padding-top: 3px;
      padding-bottom: 2px;
    }

    div.body .button {
      padding: .2rem .4rem;
      margin: .2rem;
      font-size: .75rem;
      border-radius: .2rem;
      display: inline-block;
      font-weight: 400;
      line-height: 1.5;
      color: #004B6B;
      border-color: #004B6B;
      text-align: center;
      text-decoration: none;
      vertical-align: middle;
      cursor: pointer;
      -webkit-user-select: none;
      -moz-user-select: none;
      user-select: none;
      border: 1px solid #004B6B;
      white-space: nowrap;
      }

    div.body .button:hover {
      background-color: #004B6B;
      color: white;
    }

    #title {
      font-size: 28px;
      margin: 0;
    }

    /* Allow title to wrap to delay overflow */
    @media (min-width: 1200px) and (max-width: 1264px) {
      #title {
        font-size: 28px;
        margin: 0;
        max-width: 150px;
      }
    }

    #image {
      text-align: left;
      margin: 2px;
    }

    #data {
      height: 200px;
      width: 440px;
    }

  </style>
{% endblock %}

{% block image %}
                <a href="_static/nimbleObject.png">
                  <img src="_static/nimbleObject.png" class="img-fluid" id="data">
                </a>
{% endblock %}

{% block cheatsheet %}
  <div class="container-fluid">
    <div class="row">
      <p style="font-size:1.2rem;font-weight:bold">
        <a href="_static/cheatsheet.pdf" target="_blank">
          <i class="bi bi-file-pdf"></i>Download as PDF
        </a>
      </p>
    </div>
    <br>
    <div class="row align-items-end">
      <div class="col-12 col-xl-auto">
      {% block title %}{{super()}}{% endblock %}
      </div>
      <div class="col-12 col-xl">
        {% block description %}{{super()}}{% endblock %}
      </div>
      <div class="col-12 col-xl-auto">
        <p>
          <button class="button" id="expand">expand all</button>
          <button class="button" id="collapse">collapse all</button>
        </p>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block object_head %}
        <div class="heading rounded section-head active">
          <i class="bi bi-caret-down-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="row">
          <div class="col-lg-5">
            <div class="content">
              {% block object_image %}
              {{super()}}
              {% endblock %}
            </div>
          </div>
          <div class="col-lg-7">
            <div class="content">
              {% block object_description %}
              {{super()}}
              {% endblock %}
            </div>
          </div>
        </div>
        <div class="content">
          {% block object_info %}
          {{super()}}
          {% endblock %}
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block io_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block io_data_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block io_data_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block io_fetching_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block io_fetching_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block io_saving_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block io_saving_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block info_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="content">
          {% block info_content %}
          {{super()}}
          {% endblock %}
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block visualization_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block visualization_printing_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block visualization_printing_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block visualization_plotting_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block visualization_plotting_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block iteration_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="content">
          {% block iteration_content %}
          {{super()}}
          {% endblock %}
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block querying_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block querying_data_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block querying_data_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block querying_indexing_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block querying_indexing_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block querying_querystrings_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block querying_querystrings_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block math_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block math_operators_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block math_operators_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block math_stretch_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block math_stretch_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block math_linalg_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block math_linalg_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">

        {% block statistics_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}  

        <div class="subsection">
          <div class="heading rounded subhead">
            {% block statistics_methods_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block statistics_methods_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>

        <div class="subsection">
          <div class="heading rounded subhead">
            {% block statistics_choice_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block statistics_choice_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        
      </div>
    </div>      
    <div class="row section">
      <div class="col-lg-12">
        {% block manipulation_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block manipulation_copying_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block manipulation_copying_content %}
            {{super()}}
            {% endblock %}
          </div>
          <div class="footnote">
            {{ self.manipulation_footnote() }}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block manipulation_modification_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block manipulation_modification_content %}
            {{super()}}
            {% endblock %}
          </div>
          <div class="footnote">
            {{ self.manipulation_footnote() }}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block manipulation_structural_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block manipulation_structural_content %}
            {{super()}}
            {% endblock %}
          </div>
          <div class="footnote">
            {{ self.manipulation_footnote() }}
          </div>
        </div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block submodules_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="content">{% block submodules_content %}{{super()}}{% endblock %}</div>
      </div>
    </div>
    <div class="row section">
      <div class="col-lg-12">
        {% block ml_head %}
        <div class="heading rounded section-head">
          <i class="bi bi-caret-right-fill"></i>
          {{super()}}
        </div>
        {% endblock %}
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block ml_interfaces_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block ml_interfaces_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block ml_arguments_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block ml_arguments_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block ml_trainedlearner_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block ml_trainedlearner_content %}
            {{super()}}{% endblock %}
          </div>
        </div>
        <div class="subsection">
          <div class="heading rounded subhead">
            {% block ml_training_head %}
            <i class="bi bi-caret-right-fill"></i>
            {{super()}}
            {% endblock %}
          </div>
          <div class="content">
            {% block ml_training_content %}
            {{super()}}
            {% endblock %}
          </div>
        </div>
      </div>
    </div>
  </div>
{% endblock %}

{% block rst %}
cheatsheet
==========

.. raw:: html

  {% filter indent(width=2, first=True) %}
  {{super()}}
  {% endfilter %}

{% endblock %}
